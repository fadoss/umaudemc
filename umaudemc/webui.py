#
# Web-based user interface
#

import cgi
import http.server
import json
import math
import os
import pathlib
import re
import sys
import webbrowser

from . import mproc, usermsgs, resources

# Static files are included in the package

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = 'data'
ADDRESS_REGEX = re.compile('([a-zA-Z0-9.]*):([0-9]+)')

#
# Handle the path explorer of the graphical interface
#
# The possibility of setting an explicit root to limit the access to the
# filesystem and supporting access across drives in Windows adds some
# complications. Paths seen by the browser are always absolute POSIX paths.
#


class PathHandler:
	"""Path handler for POSIX"""

	PATH_BASE = pathlib.PurePosixPath('/')

	def __init__(self):
		self.root = None
		self.default = None

	def set_root(self, root):
		self.root = pathlib.Path(root).resolve()

		if not self.root.exists():
			usermsgs.print_error('The given root directory does not exist.')
			return False

		return True

	def set_default(self, default):
		# The default directory by default
		self.default = self.root if self.root else pathlib.Path.cwd()

		if default:
			default_dir = pathlib.Path(default).resolve()

			if not default_dir.exists():
				usermsgs.print_warning('The given source directory does not exist. Ignoring it.')

			elif self.root and self.root != default_dir and \
				self.root not in default_dir.parents:
				usermsgs.print_warning('The source directory must be inside the root directory. Ignoring it.')
			else:
				self.default = default_dir

		return True

	def translate_path(self, path):
		# If no path was given, the default is used
		if not path:
			full_path = self.default
			relative_path = full_path.relative_to(self.root) if self.root else full_path

		elif self.root is None:
			full_path = pathlib.Path(path).resolve()
			relative_path = full_path

		# With an explicit root, paths are given relative to it
		else:
			full_path = (self.root / path.lstrip('/')).resolve()

			# If path are not inside the root, this function will raise
			# a ValueError
			try:
				relative_path = full_path.relative_to(self.root)

			except ValueError:
				return None, None

		# Check whether the file exists
		if not full_path.exists():
			return None, None

		return full_path, relative_path

	def browse_dir(self, path):
		full_path, out_path = self.translate_path(path)

		if full_path is None:
			return (None, ) * 4

		# If the path points to a file, assume it refers to the enclosing directory
		elif not full_path.is_dir():
			full_path = full_path.parent
			out_path = out_path.parent

		out_path = self.PATH_BASE / out_path.as_posix()

		return (out_path.as_posix(),
		        out_path.parent.as_posix(),
		        [f.name for f in os.scandir(full_path) if f.is_dir() and f.name[0] != '.'],
		        [f.name for f in os.scandir(full_path) if f.is_file()
		         and f.name[0] != '.' and f.name.endswith('.maude')])


class WinPathHandler(PathHandler):
	"""Path handler for Windows (allow accessing to other drives)"""

	def get_drives(self):
		# Get the Windows drive list (Stack Overflow question 827371)

		from ctypes import windll
		import string

		drives = []
		bitmask = windll.kernel32.GetLogicalDrives()
		for letter in string.ascii_uppercase:
			if bitmask & 1:
				drives.append(f'{letter}:')
			bitmask >>= 1

		return drives

	def translate_path(self, path):
		return super().translate_path(path.lstrip('/'))

	def browse_dir(self, path):
		if self.root is None and path == '/':
			# The root in Windows is the list of drives
			return '/', '/', self.get_drives(), []
		else:
			return super().browse_dir(path + '/')


class ConnectionInfo:
	"""Persisent server information"""

	def __init__(self):
		self.path_handler = WinPathHandler() if sys.platform == 'win32' else PathHandler()
		self.remote = None
		# Initial data of the problem while waiting for the reply
		self.problem_data = {}


class RequestHandler(http.server.BaseHTTPRequestHandler):
	"""Object charged of handling the requests"""

	STATIC_FILES = {
		'/'		: ('select.htm', 'text/html; charset=utf-8'),
		'/smcview.css'	: ('smcview.css', 'text/css; charset=utf-8'),
		'/smcview.js'	: ('smcview.js', 'text/javascript; charset=utf-8'),
		'/smcgraph.js'	: ('smcgraph.js', 'text/javascript; charset=utf-8'),
	}

	def do_GET(self):
		static_value = self.STATIC_FILES.get(self.path, None)

		if static_value is not None:
			filename, content = static_value
			self.send_response(200)
			self.send_header('Content-Type', content)
			self.end_headers()

			self.wfile.write(resources.get_resource_binary(filename))
		else:
			self.send_error(404)

	def do_POST(self):
		form = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
		                        environ={'REQUEST_METHOD': 'POST',
		                                 'CONTENT_TYPE': self.headers['Content-Type']})

		question = form.getvalue('question')
		url = form.getvalue('url', None)

		if question == 'ls':
			base, parent, dirs, files = self.server.info.path_handler.browse_dir(url)

			if base is None:
				self.send_error(400)
			else:
				self.send_response(200)
				self.send_header('Content-Type', 'text/json; charset=utf-8')
				self.end_headers()

				response = {
					'base'		: base,
					'parent'	: parent,
					'dirs'		: dirs,
					'files'		: files
				}
				self.wfile.write(json.dumps(response).encode('utf-8'))
		elif question == 'sourceinfo':
			self.send_response(200)
			self.send_header('Content-Type', 'text/json; charset=utf-8')
			self.end_headers()
			if self.server.info.remote is not None:
				self.server.info.remote.shutdown()
			self.server.info.remote = mproc.MaudeRemote()
			real_path, relative_path = self.server.info.path_handler.translate_path(url)
			self.server.info.remote.load(str(real_path))
			response = {
				'modules': [{'name': name, 'type': mtype} for name, mtype in self.server.info.remote.getModules()]
			}
			self.wfile.write(json.dumps(response).encode('utf-8'))
		elif question == 'modinfo':
			self.send_response(200)
			self.send_header('Content-Type', 'text/json; charset=utf-8')
			self.end_headers()
			mod = form.getvalue('mod')
			minfo = self.server.info.remote.getModuleInfo(mod)
			response = {
				'name'		: mod,
				'type'		: minfo['type'],
				'params'	: [],
				'valid'		: minfo['valid'],
				'stateSorts'	: minfo.get('state', []),
				'strategies'	: [self.make_signature(signat) for signat in minfo.get('strat', ())],
				'props'		: [self.make_signature(signat) for signat in minfo.get('prop', ())]
			}
			self.wfile.write(json.dumps(response).encode('utf-8'))
		elif question == 'modelcheck':
			self.send_response(200)
			self.send_header('Content-Type', 'text/json; charset=utf-8')
			self.end_headers()
			data = {
				'module':  form.getvalue('mod'),
				'initial': form.getvalue('initial'),
				'formula': form.getvalue('formula'),
				'strat':   form.getvalue('strategy'),
				'opaques': form.getvalue('opaques'),
				'passign': self.make_passign(form),
				'reward':  form.getvalue('reward', None),
			}
			result = self.server.info.remote.modelCheck(data)
			response = {
				# Ok (0) or the failure cause: the state term (1), the LTL formula (2),
				# the strategy (3), the probability assignment (5), the reward term (6),
				# or the lack of appropriate standard (4) or probabilistic (7) backends
				'status': 0 if result['ok'] else {
					'term'       : 1,
					'formula'    : 2,
					'strat'      : 3,
					'nobackend'  : 4,
					'passign'    : 5,
					'reward'     : 6,
					'nopbackend' : 7,
				}.get(result['cause'], 5),
				'logic': self.logic_name(result.get('logic'))

			}
			if result['ok']:
				# Send the reference to obtain the model-checking result
				response['mcref'] = result['ref']
				# Add the initial data to result and store it
				self.server.info.problem_data[result['ref']] = data

			self.wfile.write(json.dumps(response).encode('utf-8'))
		elif question == 'wait':
			mcref = int(form.getvalue('mcref'))

			# Model checking is actually done here
			result = self.server.info.remote.get_result(mcref)

			if result is None:
				self.send_error(500)
				return

			# Get the initial data to include it in the response
			data = self.server.info.problem_data.pop(mcref)

			self.send_response(200)
			self.send_header('Content-Type', 'text/json; charset=utf-8')
			self.end_headers()

			# JSON does not support infinity, so we translate it here
			if result['rtype'] == 'n' and math.isinf(result['result']):
				result['result'] = '∞'

			response = {
				'hasCounterexample': result['hasCounterexample'],
				'formula' : data['formula'],
				'initial' : data['initial'],
				'strat'   : data.get('strat'),
				'result'  : result['result'],
				'rtype'   : result['rtype'],
				'leadIn'  : result.get('leadIn'),
				'cycle'   : result.get('cycle'),
				'states'  : result.get('states'),
				'passign' : data['passign'],
				'reward'  : data['reward'],
			}
			self.wfile.write(json.dumps(response).encode('utf-8'))
		elif question == 'cancel':
			self.server.info.remote.shutdown()
			self.server.info.remote = mproc.MaudeRemote()

			self.send_response(200)
			self.send_header('Content-Type', 'text/plain; charset=utf-8')
			self.end_headers()

			self.wfile.write('done'.encode('utf-8'))
		else:
			self.send_error(404)

	@staticmethod
	def logic_name(name):
		"""User-friendly name of each logic"""
		return 'μ-calculus' if name == 'Mucalc' else name

	@staticmethod
	def make_signature(signat):
		head, *itypes = signat
		return {'name': head, 'params': itypes}

	@staticmethod
	def make_passign(form):
		"""Build a passign term from its name and argument"""

		name = form.getvalue('pmethod', None)

		if name is None:
			return None

		argument = form.getvalue('pargument', '')

		# Add mdp- prefix when MDP is selected and admissible
		can_mdp = name not in ('uaction', 'strategy')
		mdp = 'mdp-' if can_mdp and form.getvalue('mdp') == 'true' else ''

		return f'{mdp}{name}({argument})' if argument else f'{mdp}{name}'


def run(args):
	"""Run the web interface"""

	# Use given address and port number
	server_address = ('127.0.0.1', 8000)

	if args.address is not None:
		match = ADDRESS_REGEX.fullmatch(args.address)
		if match:
			server_address = (match.group(1), int(match.group(2)))
		else:
			usermsgs.print_warning('Bad address:port specification. Ignoring it.')

	httpd = http.server.ThreadingHTTPServer(server_address, RequestHandler)
	httpd.info = ConnectionInfo()

	# Limit access to the given root path
	if args.rootdir is not None:
		if not httpd.info.path_handler.set_root(args.rootdir):
			return 1

	# The source directory is the initial directory that will be shown when Maude files
	# are opened. By default, it is the root or the current working directory.
	if not httpd.info.path_handler.set_default(args.sourcedir):
		return 1

	# Show the address and open a browser
	full_address = f'http://{server_address[0] if server_address[0] != "" else "127.0.0.1"}:{server_address[1]}'

	usermsgs.print_info(f'Serving at {full_address}')
	if not args.no_browser:
		webbrowser.open(full_address)

	try:
		httpd.serve_forever()
	except KeyboardInterrupt:
		print('Server stopped by the user')

	return 0

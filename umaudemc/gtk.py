#
# Desktop interface with Gtk (wrapper around the web interface)
#

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Gio', '2.0')
gi.require_version('WebKit', '6.0')
from gi.repository import Gtk, Gio, GLib,  WebKit

import multiprocessing


def start_server(queue):
	"""Start a server in a separate process"""
	from . import webui
	import http.server

	httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), webui.RequestHandler)
	httpd.info = webui.ConnectionInfo()
	httpd.info.path_handler.set_default(None)
	queue.put(httpd.server_port)
	httpd.serve_forever()


class ModelCheckerWindow(Gtk.ApplicationWindow):
	"""The main window of the model checker GUI"""

	def __init__(self, port=8000):
		super().__init__(title='Unified Maude model-checking tool')
		self.connect('close-request', self.destroyed)
		self.set_default_size(800, 600)

		# Use a WebView as the only content of the window
		self.canvas = WebKit.WebView.new()
		self.set_child(self.canvas)
		# Enable inspector (for debugging purposes)
		self.canvas.get_settings().props.enable_developer_extras = True
		# Remove the header bar (there is already a window title)
		user_manager = self.canvas.get_user_content_manager()
		style = WebKit.UserStyleSheet('header { display: none; }',
		                              WebKit.UserContentInjectedFrames.TOP_FRAME,
		                              WebKit.UserStyleLevel.AUTHOR, None, None)
		user_manager.add_style_sheet(style)
		# Allow using the system open dialog
		self.enable_open_dialog(user_manager)

		# Load the input page
		self.canvas.load_uri(f'http://127.0.0.1:{port}/')

	def destroyed(self, window):
		"""When the window is closed"""
		self.get_application().server_process.terminate()

	def enable_open_dialog(self, user_manager):
		"""Enable access to the open dialog in the browser"""

		# Register a script so that the native open dialog is used
		# instead of the web-based file explorer
		script = WebKit.UserScript('loadSource = loadSourceNative',
		                           WebKit.UserContentInjectedFrames.TOP_FRAME,
		                           WebKit.UserScriptInjectionTime.END, None, None)
		user_manager.add_script(script)

		# Create a file open dialog
		self.dialog = self.make_open_dialog()

		# Register the custom umaudemc scheme to open the file dialog
		self.canvas.get_context().register_uri_scheme('umaudemc', self._umaudemc_request, None)
		# Allow cross-origin requests to umaudemc (from localhost)
		self.canvas.set_cors_allowlist(['umaudemc://*/*'])
		# Disable catching (for the umaudemc request to be repeatable)
		self.canvas.get_context().set_cache_model(WebKit.CacheModel.DOCUMENT_VIEWER)

	def make_open_dialog(self):
		"""Build the Maude file open dialog"""

		opendialog = Gtk.FileDialog()
		opendialog.props.title = 'Choose a Maude file'

		filter_maude = Gtk.FileFilter()
		filter_maude.set_name('Maude files (*.maude)')
		filter_maude.add_pattern('*.maude')

		filters = Gio.ListStore()
		filters.append(filter_maude)
		opendialog.props.filters = filters

		return opendialog

	def _umaudemc_request(self, request, data):
		path = request.get_uri()

		if path == 'umaudemc://open/':
			self.dialog.open(self, None, self._file_selected, request)

		else:
			request.finish_error(GLib.Error.new_literal(1, "Not found", 404))

	def _file_selected(self, dialog, result, request):
		"""Callback on the selection of a file"""
		try:
			content = dialog.open_finish(result).get_path().encode('utf-8')
			request.finish(Gio.MemoryInputStream.new_from_data(content), len(content), 'text/plain')
		except GLib.Error:
			request.finish_error(GLib.Error.new_literal(1, "Not found", 404))


def run_gtk():
	GLib.set_prgname('umaudemc')
	app = Gtk.Application.new('es.ucm.maude.umaudemc', Gio.ApplicationFlags.FLAGS_NONE)

	def window_init(app):
		# Start the web server
		queue = multiprocessing.Queue()
		app.server_process = multiprocessing.Process(target=start_server, args=(queue,))
		app.server_process.start()

		# The random server listening port is obtained through the queue
		port = queue.get()
		window = ModelCheckerWindow(port)

		app.add_window(window)
		window.show()

	app.connect('activate', window_init)
	return app.run()


if __name__ == "__main__":
	run_gtk()

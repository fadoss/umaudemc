#
# Desktop interface with Gtk
#

import threading
from xml.sax.saxutils import escape

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gio', '2.0')
gi.require_version('WebKit2', '4.0')
from gi.repository import Gtk, Gio, GLib, GObject, WebKit2

from . import resources
from . import mproc


MODULE_TYPE_NAMES = {
	'fmod'	: 'Functional module',
	'mod'	: 'System module',
	'smod'	: 'Strategy module',
	'fth'	: 'Functional theory',
	'th'	: 'System theory',
	'sth'	: 'Strategy theory'
}


class Banner(Gtk.Bin):
	"""Banner where module and counterexample information is presented"""

	__gsignals__ = {
		# A strategy or atomic proposition name has been clicked
		'hint-clicked': (GObject.SignalFlags.RUN_FIRST, None, (bool, str, int))
	}

	def __init__(self):
		# Three widgets may appear in the banner (only one at a time) to respectively
		# show a fixed message, module information or a counterexample

		self.label = Gtk.Label(
			label='Please select a Maude file and a Maude module defining the system and properties specification',
			use_markup=True,
			justify=Gtk.Justification.CENTER,
			track_visited_links=False
		)
		self.grid = None
		self.canvas = None

		Gtk.Bin.__init__(self, child=self.label)

	def __makeGrid(self, info):
		"""Compose the grid where module information is presented"""

		# The grid has 2 columns and 4 rows starting with a header with the module
		# name and type, and followed by some attributes of the module whose values
		# appear in the second column

		self.module_type = Gtk.Label(label=f'<big>{MODULE_TYPE_NAMES[info["type"]]}</big>', use_markup=True)
		self.module_name = Gtk.Label(label=f'<big>{info["name"]}</big>', use_markup=True)

		# Labels for each module attribute name
		sort_label = Gtk.Label.new('State sort:')
		props_label = Gtk.Label.new('Atomic propositions:')
		self.strats_label = Gtk.Label.new('Strategies:')

		# Labels for each module attribute value (to be filled on its selection)
		self.sort_value = Gtk.Label.new('')
		self.props_value = Gtk.Label(label='', use_markup=True, track_visited_links=False)
		self.strats_value = Gtk.Label(label='', use_markup=True, track_visited_links=False)
		self.props_value.set_line_wrap(True)
		self.strats_value.set_line_wrap(True)

		# Propositions and strategies can be clicked to be filled
		# in their corresponding fields
		self.props_value.connect('activate-link', self.hint_clicked)
		self.strats_value.connect('activate-link', self.hint_clicked)

		# Align labels to the right and values to the left
		self.module_type.set_halign(Gtk.Align.END)
		sort_label.set_halign(Gtk.Align.END)
		props_label.set_halign(Gtk.Align.END)
		self.strats_label.set_halign(Gtk.Align.END)
		self.module_name.set_halign(Gtk.Align.START)
		self.sort_value.set_halign(Gtk.Align.START)
		self.props_value.set_halign(Gtk.Align.START)
		self.strats_value.set_halign(Gtk.Align.START)

		self.grid = Gtk.Grid.new()
		self.grid.set_valign(Gtk.Align.CENTER)
		self.grid.set_halign(Gtk.Align.CENTER)
		self.grid.set_column_spacing(15)
		self.grid.set_row_spacing(5)
		self.grid.attach(self.module_type, 0, 0, 1, 1)
		self.grid.attach(self.module_name, 1, 0, 1, 1)
		self.grid.attach(sort_label, 0, 1, 1, 1)
		self.grid.attach(self.sort_value, 1, 1, 1, 1)
		self.grid.attach(props_label, 0, 2, 1, 1)
		self.grid.attach(self.props_value, 1, 2, 1, 1)
		self.grid.attach(self.strats_label, 0, 3, 1, 1)
		self.grid.attach(self.strats_value, 1, 3, 1, 1)

		self.grid.show_all()

	def __set_current(self, widget):
		"""Set the current widget in the banner"""

		if self.get_child() != widget:
			self.remove(self.get_child())
			self.add(widget)

	@staticmethod
	def composeSignature(prefix, signature):
		"""Create a link with the signature of an atomic proposition or strategy"""

		# Displayed signature
		composed = escape(signature[0]) if len(signature) == 1 else \
			'{}({})'.format(escape(signature[0]), ', '.join(signature[1:]))

		return f'<a href="#{prefix}:{escape(signature[0])}:{len(signature) - 1}">{composed}</a>'

	def show_modinfo(self, info):
		"""Process and presents the given module information"""

		if not info['valid']:
			self.label.set_label(f'<big>{MODULE_TYPE_NAMES[info["type"]]}  {info["name"]}</big>'
			                     '\n\nNot valid for model checking')

			self.__set_current(self.label)
		else:
			if self.grid is None:
				self.__makeGrid(info)

			self.module_type.set_label(f'<big>{MODULE_TYPE_NAMES[info["type"]]}</big>')
			self.module_name.set_label(f'<big>{info["name"]}</big>')

			# State sort subtypes
			self.sort_value.set_label(' '.join(info['state']))

			# Atomic propositions
			if info['prop']:
				self.props_value.set_label(' '.join(map(lambda x: self.composeSignature('p', x), info['prop'])))
			else:
				self.props_value.set_label('none')

			# Strategies
			if info['strat']:
				self.strats_value.set_label(' '.join(map(lambda x: self.composeSignature('s', x), info['strat'])))
				self.strats_value.set_visible(True)
				self.strats_label.set_visible(True)
			else:
				self.strats_value.set_visible(False)
				self.strats_label.set_visible(False)

			self.__set_current(self.grid)

	def hint_clicked(self, banner, uri):
		"""A link was clicked"""

		if len(uri) == 0 or uri[0] != '#':
			return True

		# The URI contains the type of objects, its name and arity
		kind, name, arity = uri[1:].split(':')
		self.emit('hint-clicked', kind == 's', name, int(arity))

		return True

	@classmethod
	def dict2jsobj(cls, value):
		"""Convert a Python dictionary to a Javascript object literal"""

		if isinstance(value, bool):
			return 'true' if value else 'false'

		elif value is None:
			return 'null'

		elif isinstance(value, dict):
			entries = [f'{key}: {cls.dict2jsobj(value)}' for key, value in value.items()]
			return '{' + ', '.join(entries) + '}'

		elif isinstance(value, list):
			return '[' + ', '.join(cls.dict2jsobj(elem) for elem in value) + ']'

		else:
			return repr(value)

	def show_counterexample(self, result, background=None):
		"""Show a counterexample in the banner"""

		js_result = self.dict2jsobj(result)

		if self.canvas is None:
			self.canvas = WebKit2.WebView.new()
			# Enable inspector (for debugging purposes)
			self.canvas.get_settings().props.enable_developer_extras = True
			# Resolve files with the umaudemc scheme to static resources
			self.canvas.get_context().register_uri_scheme('umaudemc', self.get_webview_resource, None)
			self.html_load_id = self.canvas.connect('load-changed', self.html_load, f'initCanvas({js_result})')
			# Load the input data page
			self.canvas.load_html(resources.get_resource_content('result.htm'))
			self.canvas.show_all()
		else:
			self.canvas.run_javascript(f'initCanvas({js_result})', None, self.js_callback, None)

		# Set the background color to that of the windows (if this is background)
		if background is not None:
			self.canvas.set_background_color(background)

		# Replace the current widget with this
		self.__set_current(self.canvas)

	def set_editable(self, value):
		"""Disable active widgets in the banner"""

		if self.grid is not None:
			self.grid.set_sensitive(value)

	def get_webview_resource(self, request, data):
		"""Get a resource from the package data"""

		path = request.get_path()

		if path == 'smcview.css':
			content = resources.get_resource_binary(path)
			request.finish(Gio.MemoryInputStream.new_from_data(content), len(content), 'text/css')

		elif path == 'smcgraph.js':
			content = resources.get_resource_binary(path)
			request.finish(Gio.MemoryInputStream.new_from_data(content), len(content), 'text/javascript')

		else:
			request.finish_error(GLib.Error.new_literal(GLib.FileError(GLib.FileError.NOENT)))

	def js_callback(self, sender, result, data):
		"""Callback for run_javascript"""

		self.canvas.run_javascript_finish(result)

	def html_load(self, canvas, event, call):
		"""Callback when the page has been loaded"""

		if event == WebKit2.LoadEvent.FINISHED:
			canvas.run_javascript(call, None, self.js_callback, None)
			canvas.disconnect(self.html_load_id)


class ModelCheckerWindow(Gtk.ApplicationWindow):
	"""The main window of the model checker GUI"""

	__gsignals__ = {
		# Model checking has finished
		'mc-finished': (GObject.SignalFlags.RUN_FIRST, None, (object, ))
	}

	def __init__(self):
		super().__init__(title='Maude strategy-aware model checker')
		self.connect('delete-event', self.destroyed)
		self.set_border_width(6)
		self.resize(800, 600)

		# The window consists of a header (top), a banner (self.vbox). and two
		# bottom containers (bottom1 and bottom2) vertically arranged

		self.vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
		top = Gtk.Box(spacing=6)
		self.banner = Banner()
		self.banner.connect('hint-clicked', self.link_click)
		bottom1 = Gtk.Box(spacing=6)
		bottom2 = Gtk.Box(spacing=6)
		self.vbox.pack_start(top, False, False, 0)
		self.vbox.pack_start(self.banner, True, True, 0)
		self.vbox.pack_start(bottom1, False, False, 0)
		self.vbox.pack_start(bottom2, False, False, 0)
		self.add(self.vbox)

		# A header allow choosing and show the current file and module
		self.openbutton = Gtk.FileChooserButton.new_with_dialog(self.make_open_dialog())
		self.add_labeled_widget(top, 'Maude file:', self.openbutton)
		self.openbutton.connect('file-set', self.file_selected)

		self.module = Gtk.ComboBoxText()
		self.add_labeled_widget(top, '_Module:', self.module)
		self.module.connect('changed', self.module_selected)

		# A footer includes entries to input the problem data
		self.initial = Gtk.Entry()
		self.add_labeled_widget(bottom1, '_Initial term:', self.initial)

		self.formula = Gtk.Entry()
		self.add_labeled_widget(bottom1, '_Formula:', self.formula)

		self.sexpr = Gtk.Entry()
		self.add_labeled_widget(bottom2, '_Strategy:', self.sexpr)

		self.opaques = Gtk.Entry()
		self.add_labeled_widget(bottom2, '_Opaque strategies:', self.opaques)

		# A button starts model checking
		self.button = Gtk.Button(label='Model check', sensitive=False)
		bottom2.pack_start(self.button, False, False, 0)
		self.button.connect('clicked', self.model_check_clicked)
		self.connect('mc-finished', self.model_check_finished)

		# When an entry changes, the usability of the model-checking
		# button is updated
		self.initial.connect('changed', self.entry_changed)
		self.formula.connect('changed', self.entry_changed)

		# Widgets lists (for doing actions on all of them)
		self.entry_widgets = [self.openbutton, self.module, self.initial,
		                      self.formula, self.sexpr, self.opaques]
		self.text_widgets = [self.initial, self.formula, self.sexpr, self.opaques]

		# Whether widgets are editable (not blocked by other operation)
		self.editable = True
		# A connection to an interruptable Maude session
		self.maude_session = mproc.MaudeRemote()

	def signal_entry_error(self, entry, value):
		"""Mark an entry widget with a warning in case of syntax error"""

		entry.set_icon_from_icon_name(Gtk.EntryIconPosition.SECONDARY,
		                              'dialog-warning' if value else None)

	def add_labeled_widget(self, container, label, widget, expand=True):
		"""Add a widget together with a label to a container"""

		label_widget = Gtk.Label.new_with_mnemonic(label)
		label_widget.set_mnemonic_widget(widget)
		container.pack_start(label_widget, False, False, 0)
		container.pack_start(widget, expand, expand, 0)

	def make_open_dialog(self):
		"""Build the Maude file open dialog"""

		opendialog = Gtk.FileChooserDialog(title='Choose a Maude file',
		                                   parent=self,
		                                   action=Gtk.FileChooserAction.OPEN)

		opendialog.add_button(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL)
		opendialog.add_button(Gtk.STOCK_OPEN, Gtk.ResponseType.OK)

		filter_maude = Gtk.FileFilter()
		filter_maude.set_name('Maude files (*.maude)')
		filter_maude.add_pattern('*.maude')
		opendialog.add_filter(filter_maude)

		return opendialog

	def destroyed(self, event, other):
		"""Callback for the window destruction"""

		self.maude_session.shutdown()
		return False

	def entry_changed(self, editable):
		"""Callback when some entry has changed (to disable/enable the MC button)"""

		complete = self.initial.get_text() and self.formula.get_text() \
		           and self.openbutton.get_filename() and self.module.get_active_id()
		self.button.set_sensitive(complete)

	def file_selected(self, data):
		"""A file has been selected to be loaded"""

		filename = self.openbutton.get_filename()
		print('Loading', filename)

		if not self.maude_session.load(filename):
			self.notify_error(f'Error loading the file {filename}.')
		else:
			last_id = None
			for name, mtype in self.maude_session.getModules():
				self.module.append(name, f'{name} ({mtype})')
				last_id = name

			self.module.set_active_id(last_id)
			self.entry_changed(self.openbutton)

	def file_reload(self):
		"""Reload the currently selected file (after an interrupted session)"""

		filename = self.openbutton.get_filename()
		print('Reloading', filename)

		if not self.maude_session.load(filename):
			self.notify_error(f'Error reloading the file {filename}.')
		else:
			current_id = self.module.get_active_id()

			for name, mtype in self.maude_session.getModules():
				self.module.append(name, f'{name} ({mtype})')

			self.module.set_active_id(current_id)

	def module_selected(self, data):
		"""Callback for the selection of a module"""

		self.module_info = self.maude_session.getModuleInfo(self.module.get_active_id())
		self.show_modinfo()

	def link_click(self, sender, is_strat, text, arity):
		"""Callback for the activation of a banner link"""

		# An entry file is completed with the clicked object
		entry = self.sexpr if is_strat else self.formula

		entry.delete_selection()
		inserted_text = text + ('' if arity == 0 else '(')
		entry.insert_text(inserted_text, entry.get_position())
		entry.set_position(entry.get_position() + len(inserted_text))

	def model_check_clicked(self, sender):
		"""Callback for the model-checking/cancel button"""

		# The button acts as a cancel when model checking
		if not self.editable:
			self.set_editable(True)
			self.maude_session.shutdown()
			self.maude_session = mproc.MaudeRemote()
			self.file_reload()
			return

		# Pass the model checking data to Maude
		data = {
			'file'		: self.openbutton.get_filename(),
			'module'	: self.module.get_active_id(),
			'initial'	: self.initial.get_text(),
			'strat'		: self.sexpr.get_text(),
			'formula'	: self.formula.get_text(),
			'opaques'	: self.opaques.get_text().split()
		}

		info = self.maude_session.modelCheck(data)

		# Model checking has not been done yet, but the input data of the problem
		# has been checked. info includes a reference to obtain the actual result.

		if not info['ok']:
			cause = info['cause']
			self.signal_entry_error(self.initial, cause == 'term')
			self.signal_entry_error(self.formula, cause == 'formula')
			self.signal_entry_error(self.sexpr, cause == 'strat')
			self.signal_entry_error(self.opaques, cause == 'opaque')

			if cause == 'nobackend':
				self.notify_error(f'No installed backend for the {info["logic"]} logic')

		else:
			# Hide previous error marks
			for widget in self.text_widgets:
				self.signal_entry_error(widget, False)

			self.set_editable(False)

			# Model checking is started in a separate thread not to block the
			# interface (although most components are disabled)
			self.thread = threading.Thread(target=self.model_check_action, args=(info['ref'], data, ))
			self.thread.daemon = True
			self.thread.start()

	def notify_error(self, text):
		"""Notify an error to the user (with a dialog)"""

		dialog = Gtk.MessageDialog(text=text,
		                           buttons=Gtk.ButtonsType.OK,
		                           message_type=Gtk.MessageType.ERROR)
		dialog.run()
		dialog.destroy()

	def set_editable(self, value):
		"""Enable or disable the active components of the window"""

		self.editable = value

		for widget in self.entry_widgets:
			widget.set_sensitive(value)
		self.banner.set_editable(value)

		self.button.set_label('Model check' if value else 'Cancel')

	def model_check_action(self, mcref, data):
		"""Callback for the result of the model-checking task"""

		# This is not run in the GTK thread

		result = self.maude_session.get_result(mcref)

		# Complement result with the initial data
		result['formula'] = data['formula']
		result['initial'] = data['initial']
		result['strat'] = data.get('strat')

		def emit_mcfinished():
			# Instead of emitting the signal, its handler code can be done here
			self.emit('mc-finished', result)
			return False

		GLib.idle_add(emit_mcfinished)

	def model_check_finished(self, sender, result):
		"""Callback in the GTK thread for the model checking finalization"""

		# Join the model checker thread (just finished)
		self.thread.join(2)

		holds = result['result']
		has_counterexample = result['hasCounterexample']

		if not has_counterexample or holds:
			# InfoBar will look more like the web version
			dialog = Gtk.MessageDialog(text=('✔ The property holds.' if holds else '✗ The property does not hold.'),
			                           buttons=Gtk.ButtonsType.OK)
			dialog.run()
			dialog.destroy()

		bg = self.get_style_context().get_background_color(self.get_state())
		if has_counterexample:
			self.banner.show_counterexample(result, background=bg)
		self.set_editable(True)

	def show_modinfo(self):
		"""Set the module information in the banner"""

		self.banner.show_modinfo(self.module_info)


def run_gtk():
	GLib.set_prgname('umaudemc')
	app = Gtk.Application.new('es.ucm.maude.umaudemc', Gio.ApplicationFlags.FLAGS_NONE)

	def window_init(app):
		window = ModelCheckerWindow()
		app.add_window(window)
		window.show_all()

	app.connect('activate', window_init)
	return app.run()

if __name__ == "__main__":
	run_gtk()

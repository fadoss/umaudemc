function loadSource()
{
	var sourceFile = document.getElementById('sourceFile')
	document.getElementById('openFile').style.display = 'flex'

	browseDir(sourceFile.fullPath ? sourceFile.fullPath : '', 'source')
}

function closeOpenFileDialog()
{
	document.getElementById('openFile').style.display = 'none'
}

function browseDir(dir)
{
	const request = new XMLHttpRequest()

	request.onreadystatechange = function()
	{
		if (this.readyState == XMLHttpRequest.DONE && this.status == 200)
		{
			var listing = JSON.parse(this.responseText)

			// Shows the files as a unordered list
			var fileList = document.getElementById('of-fileList')
			var out = ''

			if (listing.base != '/')
				out += `<li class="of-item" onclick="browseDir('${listing.parent}')">[Parent]</li>`

			for (dir of listing.dirs)
				out += `<li class="of-item" onclick="browseDir('${listing.base}/${dir}')">📁 ${dir}</li>`

			for (file of listing.files)
				out += `<li class="of-item" onclick="openFile('${listing.base}/${file}')">${file}</li>`

			fileList.innerHTML = out
		}
	}

	var question = new FormData()

	question.append('question', 'ls')
	question.append('url', dir)

	request.open('post', 'ask')
	request.send(question)
}

function openFile(file)
{
	closeOpenFileDialog()

	var sourceFile = document.getElementById('sourceFile')
	sourceFile.innerText = file.split('/').pop()
	sourceFile.fullPath = file
	loadSourceModules(file)
}

function buttonToggle()
{
	document.getElementById('send').disabled = document.getElementById('initial').value == ''
		|| document.getElementById('formula').value == ''
		|| document.getElementById('module').disabled
		|| !document.getElementById('description').db.valid
}

function disableInput(disabled)
{
	document.getElementById('send').disabled = disabled
	document.getElementById('initial').disabled = disabled
	document.getElementById('formula').disabled = disabled
	document.getElementById('module').disabled = disabled
	document.getElementById('module').disabled = disabled
	document.getElementById('strategy').disabled = disabled
	document.getElementById('opaques').disabled = disabled
	document.getElementById('sourceFileOpen').disabled = disabled
	document.getElementById('description').disabled = disabled
}

function loadSourceModules(file)
{
	const request = new XMLHttpRequest()
	const smodule = document.getElementById('module')

	request.onreadystatechange = function()
	{
		if (this.readyState == XMLHttpRequest.DONE && this.status == 200)
		{
			var listing = JSON.parse(this.responseText);

			for (module of listing.modules)
				if (module.type != 'fmod' && module.type != 'fth')
				{
					var option = new Option(module.name + ' (' + module.type + ')', module.name)
					smodule.options.add(option)
				}

			// Update graphical components
			smodule.selectedIndex = smodule.options.length - 1
			smodule.disabled = false
			loadModule()
		}
	}

	// Discard modules from previous files
	smodule.options.length = 0

	var question = new FormData()
	question.append('question', 'sourceinfo')
	question.append('url', file)
	request.open('post', 'ask')
	request.send(question)
}

function addPropToFormula(prop)
{
	document.getElementById('formula').value += prop
	buttonToggle()
}

function setStrategy(strategy)
{
	document.getElementById('strategy').value = strategy
	buttonToggle()
}

function formatSignature(signature)
{
	var displayName = signature.name

	if (signature.params.length > 0)
	{
		displayName += '(' + signature.params[0]
		for (i = 1; i < signature.params.length; i++)
			displayName += ',' + signature.params[i]
		displayName += ')'
	}

	return displayName
}

function loadModule()
{
	const request = new XMLHttpRequest()
	const currentModule = document.getElementById('module').value
	var description = document.getElementById('description')

	request.onreadystatechange = function()
	{
		if (this.readyState == XMLHttpRequest.DONE && this.status == 200)
		{
			var listing = JSON.parse(this.responseText)
			description.db = listing

			var text = ''

			switch (listing.type) {
				case 'fmod' : text += 'Functional module '; break
				case 'mod' : text += 'System module '; break
				case 'fth' : text += 'Functional theory '; break
				case 'th' : text += 'System theory '; break
				case 'smod' : text += 'Strategy module '; break
				case 'sth' : text += 'Strategy theory '; break
			}

			if (listing.params.length != 0)
				text = 'Parameterized ' + text.toLowerCase()

			text = `<span style="font-size: 110%;">${text} <code>${currentModule}`

			// Adds theory parameters
			if (listing.params.length > 0)
			{
				text += '{' + listing.params[0]
				for (i = 0; i < listing.params.length; i++)
					text += ',' + listing.params[i]
				text += '}'
			}

			text += '</code></span>'

			// If the module is valid for model checking, shows
			// some relevant information about it
			if (listing.valid)
			{
				// State sort
				text += '<table class="sumtable"><tr><td>State sort:</td><td>'

				if (listing.stateSorts.length == 0)
					text += ' State'

				for (msort of listing.stateSorts)
					text += ' ' + msort

				// Atomic propositions
				text += '</td></tr><tr><td>Atomic propositions:</td><td>'

				if (listing.props.length == 0)
					text += ' no atomic propositions defined'

				for (prop of listing.props)
					text += ` <span onclick="addPropToFormula('${prop.name}')">${formatSignature(prop)}</span>`

				text += '</td></tr>'

				// Strategies
				if (listing.type == 'smod' || listing.type == 'sth')
				{
					text += '<tr><td>Strategies:</td><td>'

					if (listing.strategies.length == 0)
						text += ' no strategies defined'

					for (strat of listing.strategies)
						text += ` <span onclick="setStrategy('${strat.name}')">${formatSignature(strat)}</span>`

					text += '</td></tr>'
				}

				text += '</table>'
			}
			else
			{
				text += 'Not valid for model checking.'
			}

			buttonToggle()
			description.innerHTML = text
		}
	}


	description.innerText = 'Please select a Maude file and a Maude module defining the system and properties specification.'

	var question = new FormData()
	question.append('question', 'modinfo')
	question.append('mod', currentModule)
	request.open('post', 'ask')
	request.send(question)
}

function modelcheck()
{
	var request = new XMLHttpRequest()
	var textArea = document.getElementById('source')

	request.onreadystatechange = function()
	{
		if (this.readyState == XMLHttpRequest.DONE && this.status == 200)
		{
			var listing = JSON.parse(this.responseText)
			var errbar = document.getElementById('errorbar')

			switch (listing.status)
			{
				case 0 : errbar.innerHTML = 'Waiting for the model checker... or <a class="cancelButton" onclick="cancelChecking()">cancel</a> it'; break
				case 1 : errbar.innerText = 'Syntax error at the initial term'; break
				case 2 : errbar.innerText = 'Syntax error at the formula'; break
				case 3 : errbar.innerText = 'Syntax error at strategy expression'; break
				case 4 : errbar.innerText = `No installed backends for the ${listing.logic} logic`; break
			}

			errbar.style.display = 'block'

			if (listing.status == 0)
				waitModelChecker(listing.mcref)
		}
	}

	var question = new FormData()

	question.append('question', 'modelcheck')
	question.append('mod', document.getElementById('module').value)
	question.append('initial', document.getElementById('initial').value)
	question.append('formula', document.getElementById('formula').value)
	question.append('strategy', document.getElementById('strategy').value)
	question.append('opaques', document.getElementById('opaques').value)

	request.open('post', 'ask')
	request.send(question)
}

function escapeHTMLChars(text) {
	return text.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;")
		.replace(/'/g, "&#039;")
}

function waitModelChecker(mcref) {
	var request = new XMLHttpRequest()

	request.onreadystatechange = function()
	{
		if (this.readyState == XMLHttpRequest.DONE && this.status == 200)
		{
			var listing = JSON.parse(this.responseText)
			var errbar = document.getElementById('errorbar')

			// Enable input
			disableInput(false)

			if (listing.holds)
			{
				errbar.innerHTML = `The property <span class="baremph">${escapeHTMLChars(listing.formula)}</span> holds from <span class="baremph">${escapeHTMLChars(listing.initial)}</span>` + (listing.strat == '' ? '' : ` using <span class="baremph">${escapeHTMLChars(listing.strat)}</span>`)
				errbar.style.display = 'block'
			}
			else if (!listing.hasCounterexample)
			{
				errbar.innerHTML = `The property <span class="baremph">${escapeHTMLChars(listing.formula)}</span> does not hold from <span class="baremph">${escapeHTMLChars(listing.initial)}</span>` + (listing.strat == '' ? '' : ` using <span class="baremph">${escapeHTMLChars(listing.strat)}</span>`)
				errbar.style.display = 'block'
			}
			else
			{
				// Hide the informative bar
				errbar.style.display = 'none'

				// Show the result dialog
				document.getElementById('resultDialog').style.display = 'flex'
				var resultHeader = document.getElementById('resultHeader')
				resultHeader.innerHTML = `The property <span class="baremph">${escapeHTMLChars(listing.formula)}</span> does not hold from <span class="baremph">${escapeHTMLChars(listing.initial)}</span>` + (listing.strat == '' ? '' : ` using <span class="baremph">${escapeHTMLChars(listing.strat)}</span>`)

				// Copy the counterexample information to the canvas
				var canvas = document.getElementById('canvas')
				var graph = document.getElementById('graph')
				graph.db = {holds: false, numberStates: 5, path: listing.leadIn, cycle: listing.cycle}

				// Remove the previous graph
				graph.innerHTML = ''
				// Paints the current graph
				graph.db.states = listing.states
				paintCanvas(canvas, graph)
			}
		}
	}

	// Disable input until model checking has finished
	disableInput(true)

	var question = new FormData()

	question.append('question', 'wait')
	question.append('mcref', mcref)
	request.open('post', 'ask')
	request.send(question)
}

function cancelChecking() {
	var request = new XMLHttpRequest()
	var question = new FormData()

	request.onreadystatechange = function()
	{
		if (this.readyState == XMLHttpRequest.DONE && this.status == 200)
		{
			// The model checking session need to be restarted
			disableInput(false)
			openFile(sourceFile.fullPath)

			// Hide the informative bar
			var errbar = document.getElementById('errorbar')
			errbar.style.display = 'none'
		}
	}

	question.append('question', 'cancel')
	request.open('post', 'ask')
	request.send(question)
}

function closeResultDialog()
{
	document.getElementById('graph').db = null
	document.getElementById('resultDialog').style.display = 'none'
}

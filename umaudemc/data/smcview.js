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
			const listing = JSON.parse(this.responseText)

			// Shows the files as a unordered list
			const fileList = document.getElementById('of-fileList')
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

function incompletePassign()
{
	switch (document.getElementById('pmethod').value) {
		case 'uaction':
		case 'term':
			return document.getElementById('parg').value == ''
		case 'strategy':
			return document.getElementById('strategy').value == ''
		default:
			return false;
	}
}

function buttonToggle()
{
	document.getElementById('send').disabled = document.getElementById('initial').value == ''
		|| document.getElementById('formula').value == ''
		|| document.getElementById('module').disabled
		|| !document.getElementById('description').db.valid
		|| (document.getElementById('command').value == 'qt' && incompletePassign())
}

function setDisplay(elem, value)
{
	elem.style.display = value

	for (label of elem.labels)
		label.style.display = value
}

function pcheckToggle()
{
	const assign = document.getElementById('pmethod')
	const parg = document.getElementById('parg')
	const reward = document.getElementById('reward')

	var mdp_display = 'block'

	if (document.getElementById('command').value == 'qt')
	{
		pmethod.style.display = 'block'
		setDisplay(reward, 'block')

		switch (pmethod.value)
		{
			case 'uaction':
				parg.placeholder = 'Comma-separated list of label=weight or label.p=prob'
				parg.labels[0].innerText = 'Label weights:'
				mdp_display = 'none'
				setDisplay(parg, 'block')
				break;
			case 'term':
				parg.placeholder = 'Term on the variables L(HS), R(HS), and A(ction)'
				parg.labels[0].innerText = 'Weight term:'
				setDisplay(parg, 'block')
				break;
			case 'strategy':
				mdp_display = 'none'
			default:
				parg.value = ''
				setDisplay(parg, 'none')
		}
	}
	else
	{
		mdp_display = 'none'
		pmethod.style.display = 'none'
		setDisplay(parg, 'none')
		setDisplay(reward, 'none')
	}

	setDisplay(document.getElementById('mdp'), mdp_display)

	buttonToggle()
}

function advancedToggle() {
	var advanced = document.getElementById('advanced')
	var button = document.getElementById('expandAdvanced')

	if (advanced.style.display == 'none')
	{
		advanced.style.display = 'inline-flex'
		button.innerText = '-'
	}
	else {
		advanced.style.display = 'none'
		button.innerText = '+'
	}
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
	document.getElementById('command').disabled = disabled
	document.getElementById('pmethod').disabled = disabled
	document.getElementById('parg').disabled = disabled
	document.getElementById('reward').disabled = disabled
}

function loadSourceModules(file)
{
	const request = new XMLHttpRequest()
	const smodule = document.getElementById('module')

	request.onreadystatechange = function()
	{
		if (this.readyState == XMLHttpRequest.DONE && this.status == 200)
		{
			const listing = JSON.parse(this.responseText);

			for (module of listing.modules)
				if (module.type != 'fmod' && module.type != 'fth')
				{
					const option = new Option(module.name + ' (' + module.type + ')', module.name)
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
			const listing = JSON.parse(this.responseText)
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
				text += 'Not valid for model checking.'

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
				case 3 : errbar.innerText = 'Syntax error at the strategy expression'; break
				case 4 : errbar.innerText = `No backends for the ${listing.logic} logic installed`; break
				case 5 : errbar.innerText = 'Syntax error at the weight specification'; break;
				case 6 : errbar.innerText = 'Syntax error at the reward term'; break;
				case 7 : errbar.innerText = 'No probabilistic backends installed'; break
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

	// Quantitative model checking stuff
	if (document.getElementById('command').value == 'qt')
	{
		question.append('pmethod', document.getElementById('pmethod').value)
		question.append('pargument', document.getElementById('parg').value)
		question.append('mdp', document.getElementById('mdp').checked)
		question.append('reward', document.getElementById('reward').value)
	}

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
			const listing = JSON.parse(this.responseText)
			const errbar = document.getElementById('errorbar')

			// Enable input
			disableInput(false)

			if (listing.result == null)
			{
				errbar.innerHTML = 'An internal error of the model checker has occurred'
				errbar.style.display = 'block'
			}
			else if (listing.hasCounterexample)
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
			else {
				var head = ''

				console.log(listing)

				if (listing.reward)
				{
					if (listing.rtype == 'r')
						head = `is between <span class="baremph">${listing.result[0]}</span> and <span class="baremph">${listing.result[1]}</span>`
					else
						head = `is <span class="baremph">${listing.result}</span>`

					head = `The value of the reward <span class="baremph">${escapeHTMLChars(listing.reward)}</span> ` + head
				}
				else {
					if (listing.rtype == 'b')
						head = listing.result ? 'holds' : 'does not hold'
					else if (listing.rtype == 'n')
						head = `holds with probability <span class="baremph">${listing.result}</span>`
					else if (listing.rtype == 'r')
						head = `holds with a probability between <span class="baremph">${listing.result[0]}</span> and <span class="baremph">${listing.result[1]}</span>`

					head = `The property <span class="baremph">${escapeHTMLChars(listing.formula)}</span> ` + head
				}

				errbar.innerHTML = head + ` from <span class="baremph">${escapeHTMLChars(listing.initial)}</span>`
					+ (listing.strat == '' ? '' : ` using <span class="baremph">${escapeHTMLChars(listing.strat)}</span>`)
				errbar.style.display = 'block'
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
	const request = new XMLHttpRequest()
	const question = new FormData()

	request.onreadystatechange = function()
	{
		if (this.readyState == XMLHttpRequest.DONE && this.status == 200)
		{
			// The model checking session need to be restarted
			disableInput(false)
			openFile(sourceFile.fullPath)

			// Hide the informative bar
			const errbar = document.getElementById('errorbar')
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

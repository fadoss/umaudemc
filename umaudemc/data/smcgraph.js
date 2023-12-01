function paintState(canvas, stateNr, x, y, nr)
{
	// State information received from the server and stored in the DOM
	var state = canvas.db.states[stateNr]

	var circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle')

	circle.setAttribute('cx', x)
	circle.setAttribute('cy', y)
	circle.setAttribute('r', nr)
	circle.setAttribute('class', state.solution ? 'solution' : 'state')

	var text = document.createElementNS('http://www.w3.org/2000/svg', 'text')

	text.setAttribute('x', x)
	text.setAttribute('y', y)
	text.setAttribute('class', 'stateLabel')

	text.innerHTML = stateNr

	// All state graphical elements are gathered in a group that manages
	// metadata and events
	var group = document.createElementNS('http://www.w3.org/2000/svg', 'g')

	let showFn = showPopup(state, nr)

	group.addEventListener('mouseover', showFn)
	group.addEventListener('click', showFn)

	group.addEventListener('mouseout', function () {
		document.getElementById('state-popup').style.visibility = 'hidden'
		document.getElementById('popup-term').innerText = ''
		document.getElementById('popup-strat').innerText = ''
	})

	group.appendChild(circle)
	group.appendChild(text)
	canvas.appendChild(group)

	return circle
}

function showPopup(state, nr)
{
	return function (event) {
		document.getElementById('popup-term').innerHTML = state.term
		document.getElementById('popup-strat').innerHTML = state.strategy
		var popup = document.getElementById('state-popup')
		// Calculates the preferred size of the popup
		popup.style.top = '0px'
		popup.style.left = '0px'
		var width = popup.clientWidth
		var height = popup.clientHeight
		var x = Math.min(Math.max(event.pageX - width / 2,
			canvas.parentElement.getBoundingClientRect().x),
			canvas.parentElement.clientWidth - width)
		// Sets the popup position
		popup.style.top = `${event.pageY - height - 2*nr}px`
		popup.style.left = `${x}px`
		popup.style.visibility = 'visible'
	}
}

function transitionText(transition)
{
	if (!transition)
		return '';

	switch (transition.type)
	{
		case 0 : return 'idle' ; break
		case 1 : return transition.label ; break
		case 2 : return `opaque(${transition.label})` ; break
	}
}

function paintSelfLoop(canvas, state, transition, nr)
{
	var x = state.cx.baseVal.value
	var y = state.cy.baseVal.value

	var dx = Math.sin(0.2) * nr
	var dy = Math.cos(0.2) * nr

	var arrow = document.createElementNS('http://www.w3.org/2000/svg', 'path')

	arrow.setAttribute('d', `M${x-dx},${y-dy}C${x-nr} ${y-2 * nr},${x+nr} ${y-2 * nr},${x+dx} ${y-dy}`)
	arrow.setAttribute('class', 'selfLoopArrow')
	canvas.appendChild(arrow)

	var label = document.createElementNS('http://www.w3.org/2000/svg', 'text')

	label.setAttribute('x', x)
	label.setAttribute('y', y - 2 * nr)
	label.setAttribute('class', 'transitionLabel')
	label.innerHTML = transitionText(transition)
	canvas.appendChild(label)
}

function paintTransition(canvas, source, target, transition, nr)
{
	if (source == target)
	{
		paintSelfLoop(canvas, source, transition, nr)
		return
	}

	var x0 = source.cx.baseVal.value
	var y0 = source.cy.baseVal.value
	var x = target.cx.baseVal.value
	var y = target.cy.baseVal.value

	var length = Math.sqrt(Math.pow(x - x0, 2) + Math.pow(y - y0, 2))

	var dx = (x - x0) / length * nr
	var dy = (y - y0) / length * nr

	var angle = Math.atan(dy / dx)
	// dx != 0

	var arrow = document.createElementNS('http://www.w3.org/2000/svg', 'path')

	arrow.setAttribute('d', `M${x0+dx},${y0+dy}L${x-dx},${y-dy}`)
	arrow.setAttribute('class', 'transitionArrow')
	canvas.appendChild(arrow)

	var label = document.createElementNS('http://www.w3.org/2000/svg', 'text')

	label.setAttribute('x', (x0 + x) / 2)
	label.setAttribute('y', (y0 + y) / 2 - 5)
	label.setAttribute('class', 'transitionLabel')
	label.setAttribute('transform', `rotate(${angle * 180 / Math.PI} ${(x0 + x) / 2} ${(y0 + y) / 2 - 5})`)
	label.innerHTML = transitionText(transition)
	canvas.appendChild(label)
}

function paintCanvas(canvas, graph) {
	var width = canvas.parentNode.clientWidth
	var height = canvas.parentNode.clientHeight * 0.95

	canvas.setAttribute('width', width)
	canvas.setAttribute('height', height)

	canvas.setAttribute('viewBox', `0 0 ${width} ${height}`)

	const pathLength = graph.db.path.length
	const cycleLength = graph.db.cycle.length

	// Node radius
	var nr = 20

	// The cycle is painted as an ellipse
	var cycleRadiusY = cycleLength > 1 ? height / 2 - 2 * nr : 0
	var cycleRadiusX = cycleRadiusY
	var cycleCenterY = height / 2
	var cycleCenterX = pathLength > 0 ? (width - cycleRadiusX - 2 * nr) : (width / 2)
	// Angle between states in the cycle
	var cangle = 2 * Math.PI / cycleLength

	// The node radius is reduced in case of collision
	if (cycleLength > 1)
	{
		let maxRadiusCycle = 2/3 * Math.sin(cangle / 2) * cycleRadiusY
		if (nr > maxRadiusCycle)
			nr = maxRadiusCycle
	}

	// Saves the graphical nodes in arrays for later use
	var drawnPath = new Array(pathLength)
	var drawnCycle = new Array(cycleLength)

	// Paints the path-to-the-cycle states
	if (pathLength > 0)
	{
		// Distance between nodes
		var d = (cycleCenterX - cycleRadiusX - 2 * nr) / pathLength

		// The node radius is reduced in case of collision
		if (nr > d / 3)
			nr = d / 3

		for (var index = 0; index < pathLength; index++)
		{
			var x = 2 * nr + d * index
			drawnPath[index] = paintState(graph, graph.db.path[index], x, cycleCenterY, nr)
		}
	}

	// Paints the cycle states
	for (var index = 0; index < cycleLength; index++)
	{
		var x = cycleCenterX - cycleRadiusX * Math.cos(- index * cangle)
		var y = cycleCenterY + cycleRadiusY * Math.sin(- index * cangle)

		drawnCycle[index] = paintState(graph, graph.db.cycle[index], x, y, nr)
	}

	// Paints the path transitions
	for (i = 0; i < pathLength; i++)
	{
		var sourceState = graph.db.states[graph.db.path[i]]
		var source = drawnPath[i]
		var targetNr = i+1 == pathLength ? graph.db.cycle[0] : graph.db.path[i+1]
		var target = i+1 == pathLength ? drawnCycle[0] : drawnPath[i+1]

		var transition = sourceState.successors.find(tr => tr.target == targetNr)

		paintTransition(graph, source, target, transition, nr)
	}

	// Paints the cycle transitions
	for (i = 0; i < cycleLength; i++)
	{
		var sourceState = graph.db.states[graph.db.cycle[i]]
		var source = drawnCycle[i]
		var nextIndex = (i+1) % cycleLength
		var targetNr = graph.db.cycle[nextIndex]
		var target = drawnCycle[nextIndex]

		var transition = sourceState.successors.find(tr => tr.target == targetNr)

		paintTransition(graph, source, target, transition, nr)
	}

	// Adjusts the font size
	graph.style.fontSize = `${(nr / 20) * parseInt(window.getComputedStyle(document.body).fontSize)}px`
}

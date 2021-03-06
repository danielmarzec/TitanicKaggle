def splitName(name):
	lastName = name[:name.index(',')]
	title = name[name.index(',') + 2:]
	return title, lastName

def getTitle(name):
	first, last = splitName(name)
	if '.' not in first:
		return 'Nan'
	title = first[:first.index('.')]
	return title
import os

class preprocessor(object):
	def __init__(self):
		self.root = os.getcwd() + "/"

	def fetch_data(self, folder):
		"""
		folder: name of folder
		TODO: wrong way to compile all files together, test files should be parsed separately,
		TOFIX: read folder from main, and get filenames here
		"""
		path, output, compiled_content = self.root + folder, [], []

		for root, dirs, filenames in os.walk(path):
			for file in filenames:
				raw_content = open(os.path.join(root, f), 'r').read()
				compiled_content += raw_content






	def process(self, folder):
		"""
		folder: name of folder
		"""
		pass
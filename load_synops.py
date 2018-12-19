

class Load_synops():
	
	def __init__(self, fname):
		self.synops = {}
		with open(fname, 'r') as data_file:
			for fline in data_file:
				line = fline.strip()
				if self.line_valid(line):
					self.process_line(line)
				
	def line_valid(self, line):
		if line[:3] != '333':
			return True
		return False
	
	def process_line(self, line):
		self.synops[line[:12]] = line[-4]
				
#l = Load_synops('SYNOPs/BA_Koliba_SYNOP_2014-2016.txt')
from __future__ import division
from datetime import datetime

class Helper():
	
	def __init__(self):
		self.err_count = 0
		pass
	
	def daytime_only(self, data):
		# s2 = datetime.strptime(s, '%Y%m%d%H%M')
		for k, v in data.items():
			try:
				timest = datetime.strptime(k, '%Y%m%d%H%M')
				if timest.hour not in range(6,19):
					#print(' del hour ' + str(timest.hour))
					del data[k]
			except:
				print('daytime strip not worked ' + str(k))
				self.err_count += 1
		return data
	
	def round_time(self, dt=None, round_to=60):
		"""Round a datetime object to any time lapse in seconds
		dt : datetime.datetime object, default now.
		roundTo : Closest number of seconds to round to, default 1 minute.
		Author: Thierry Husson 2012 - Use it as you want but don't blame me.
		"""
		if dt == None : dt = datetime.datetime.now()
		seconds = (dt.replace(tzinfo=None) - dt.min).seconds
		rounding = (seconds + round_to/2) //round_to * round_to
		return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)
			
	def standardize_date(self, img_name):
		'''
		get date&time from filename, convert it to standard tstamp ==> round to hours
		'''
		date = img_name[-22:-7]
		try:
			rounded_date = str(self.round_time(datetime.datetime(int(date[0:4]),  int(date[4:6]), int(date[6:8]), int(date[9:11]),  int(date[11:13]), int(date[13:15]),  int('0000')), round_to=3600))
			standardized_date = rounded_date.replace('-','').replace(' ','').replace(':','')[:-2]	#remove seconds,cause synops dont have 'em
			return standardized_date
		except:
			pass
			return None

	def avg_synops(self, s1, s2):
		'''
		s1, s2 instances of Load_synops.synops
		return dict
		'''
		s1_keys = set(s1.keys())
		s2_keys = set(s2.keys())
		common_keys = s1_keys.intersection(s2_keys)
		print('common_keys for synops ' + str(len(common_keys)))
		avg_synops = dict()
		for key in common_keys:
			try:
				if s1[key] != s2[key]:
					avg_synops[key] = (int(s1[key]) + int(s2[key]))/2
				else:
					avg_synops[key] = s1[key]
			except:
				# len of synops is not the same, 26208 vs 26214
				pass
		return avg_synops
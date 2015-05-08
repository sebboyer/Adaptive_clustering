import csv
import numpy as np


def write_inCSV(data,header,out_put_file):
	print "Start writing"
	with open(out_put_file, 'wb') as csv_file:
			writer = csv.writer(csv_file, delimiter = ',',quoting=csv.QUOTE_MINIMAL)#, quoting = csv.QUOTE_ALL)
			if header!=-1:
				writer.writerow(header)
			for line in data: 
				writer.writerow(line)
	print "End writing"

def extractArray_fromCSV(csv_file,skip_header):
	data=list()
	with open(csv_file, 'r') as csv_file:
			reader= csv.reader(csv_file, delimiter = ',')
			if skip_header:
				next(reader,None)
			for row in reader:
				line=list()
				for x in row:
					if x=='NULL':
						line.append(0)
					else:
						line.append(float(x))
				data.append(line)
	return np.array(data)

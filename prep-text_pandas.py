#!/usr/bin/env python
"""
Tool to pre-process documents contained one or more directories, and export a document-term matrix for each directory.
"""
import os, os.path, sys, codecs
import logging as log
from optparse import OptionParser
import text.util

import numpy as np
import pandas as pd
import glob
import os

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] directory1 directory2 ...")
	parser.add_option("--df", action="store", type="int", dest="min_df", help="minimum number of documents for a term to appear", default=10)
	parser.add_option("--tfidf", action="store_true", dest="apply_tfidf", help="apply TF-IDF term weight to the document-term matrix")
	parser.add_option("--norm", action="store_true", dest="apply_norm", help="apply unit length normalization to the document-term matrix")
	parser.add_option("--minlen", action="store", type="int", dest="min_doc_length", help="minimum document length (in characters)", default=10)
	parser.add_option("-s", action="store", type="string", dest="stoplist_file", help="custom stopword file path", default=None)
	parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="output directory (default is current directory)", default=None)
	# Parse command line arguments
	(options, args) = parser.parse_args()
	if( len(args) < 1 ):
		parser.error( "Must specify at least one directory" )	
	log.basicConfig(level=20, format='%(message)s')

	if options.dir_out is None:
		dir_out = os.getcwd()
	else:
		dir_out = options.dir_out	

	# Load required stopwords
	if options.stoplist_file is None:
		stopwords = text.util.load_stopwords()
	else:
		log.info( "Using custom stopwords from %s" % options.stoplist_file )
		stopwords = text.util.load_stopwords( options.stoplist_file )

	# Process each directory
	path = r'C:\Users\Akram Al-Turk\Box Sync\Research\Topic Modeling\WOS_articles\articles_1990_present_w_abstracts.csv'
	articles = pd.read_csv(path)

	time_1 = articles[(articles['PY'] > 1989) & (articles['PY'] < 2001)]
	time_2 = articles[(articles['PY'] > 2000) & (articles['PY'] < 2009)]
	time_3 = articles[articles['PY'] > 2008]

	three_periods = [time_1, time_2, time_3]

	for time in three_periods:
		#dir_name = os.path.basename( in_path )
		docs = time['AB'].tolist()
		doc_ids = time['UT'].tolist()
		log.info( "Found %d documents to parse" % len(docs) )

		# Pre-process the documents
		log.info( "Pre-processing documents (%d stopwords, tfidf=%s, normalize=%s, min_df=%d) ..." % (len(stopwords), options.apply_tfidf, options.apply_norm, options.min_df) )
		(X,terms) = text.util.preprocess( docs, stopwords, min_df = options.min_df, apply_tfidf = options.apply_tfidf, apply_norm = options.apply_norm )
		log.info( "Created %dx%d document-term matrix" % X.shape )

		# Save the pre-processed documents
		out_prefix = 'wos_' + str(int(time['PY'].min())) + '_' + str(int(time['PY'].max()))
		text.util.save_corpus( out_prefix, X, terms, doc_ids )

# --------------------------------------------------------------

if __name__ == "__main__":
	main()

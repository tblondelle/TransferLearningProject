

def stripMetadata(source_folder, target_folder):
	"""
	Get files from the folder [source_folder], 
	Perfom the stripping
	Put the results in file (or files) in target_folder
	"""
    pass

def simplifyRating(source_folder, target_folder):
	"""
	Get files from the folder [source_folder], 
	Simplify the rating
	Put the results in file (or files) in target_folder
	"""
	pass


def keepRelevantWords(source_folder, target_folder):
	"""
	Get files from the folder [source_folder], 
	Keep Relevant word
	Put the results in file (or files) in target_folder
	"""
	pass







# Origin datafile downloaded from jmcauley.ucsd.edu/data/amazon/ in ORIGIN_FOLDER
ORIGIN_FOLDER = ""

# In this folder are files such that : 
# - every line is of the form "[original ratings]\t[original review]"
STRIPPED_METADATA_FOLDER = ""

# In this folder are files such that : 
# - every line is of the form "[new ratings]\t[original review]" with [new ratings] in {0, 1, 2}
SIMPLIFIED_RATINGS_FOLDER = ""


# In this folder are files such that : 
# - every line is of the form "[new ratings]\t[list of relevant words]"
KEPT_RELEVANT_WORDS_FOLDER = ""

stripMetadata(ORIGIN_FOLDER, STRIPPED_METADATA_FOLDER)
simplifyRating(STRIPPED_METADATA_FOLDER, SIMPLIFIED_RATINGS_FOLDER)
keepRelevantWords(SIMPLIFIED_RATINGS_FOLDER, KEPT_RELEVANT_WORDS_FOLDER)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer



def tokenize(textList):
    # Entrées :
        # textList : liste de strings de taille N
    # Sorties :
        # X : numpy array de taille Nx100
        
    
    countvectorizer = CountVectorizer(ngram_range=(2,2))    
    X_token = countvectorizer.fit_transform(textList)
    X_token = X_token.toarray()
    
    # réduction de dimension
    truncatedsvd = TruncatedSVD(n_components=100)
    X_reduced_dim = truncatedsvd.fit_transform(X_token)
    
    return(X_reduced_dim)
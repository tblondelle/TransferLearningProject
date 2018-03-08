
    Countvectorizer.fit_transform réalise 2 opérations :
    - `countvectorizer.fit(textlist)`  ne renvoie rien, associe un indice
        (un entier) à chaque mot dans la liste de strings textlist.
        Ex : si textlist1 = ['aa bb','aa cc','dd'], la fonction prépare
        l'objet à faire 'aa'-> 1, 'bb'-> 2, 'cc'-> 3, 'dd'-> 4.
    - `X_token = countvectorizer.transform(textList)` crée un array numpy de la
        forme A(i,j) = nombre de mots d'indice j dans le string i.
        Ex : si textlist2 = ['aa aa','bb cc','dd aa zz'] et si on a fait
        `countvectorizer.fit(textlist1)` (cf exemple précédent), alors

              colonne 2 = nb de mots "bb" colonne 3 = nb de mots "cc"
             colonne 1 = nb de mots "aa"| | colonne 4 = nb de mots "dd"
                                      | | | |
                                      V V V V
        la fonction renverra    M = [[2,0,0,0],  <- string 1 = 'aa aa'
                                     [0,1,1,0],  <- string 2 = 'bb cc'
                                     [1,0,0,1]]  <- string 3 = 'dd aa zz'

        Comme le mot "zz" ne faisait pas partie de textlist1 (la liste utilisée en argument de countvectorizer.fit)
        ce mot n'est associé à rien

    Rq : l'array M est une matrice sparse (majoritairement vide), c'est un type d'objet qui permet de
    ne pas stocker des tas de zéros en mémoire. Pour la transformer en array normal, on peut faire
    M.toarray(), mais le tableau ainsi crée est souvent trop gros pour être géré.
    Le mieux est d'utiliser la décomposition en valeurs singulières, cf plus loin:
    """

    # Réduction de dimension
    truncatedsvd = TruncatedSVD(n_components=n_features) # prépare à projeter les données dans un espace à n_components=100 dimensions

    X_reduced_dim = truncatedsvd.fit_transform(X_token)
    """
    Comme Countvectorizer.fit_transform, cette instruction réalise 2 opérations
    - `truncatedsvd.fit(X_token)` prépare l'objet, lui dit d'utiliser les mots
       avec la distribution de probabilité de  X_token

    - `X_reduced_dim = truncatedsvd.fit_transform(X_token)` fait la Décomposition
        en valeurs singulières (SVD), qui est l'équivelent d'une diagonalisation
        pour des matrices rectangles. On calcule U,V,D, tq :
            - U carrée, U*transposée(U) = I_m
            - D rectancle, diagonale
            - V carrée, V*transposée(V) = I_n
        On renvoie ensuite U[:n_components], la matrice U dont on a tronqué les
        coordonnées qui a concentré l'information, un peu à la manière d'une
        ACP (pour les maths, cf Wikipédia).
    """

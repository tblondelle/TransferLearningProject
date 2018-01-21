class DataLoader():
    def __init__(self,data_directory):
        self.data_directory = data_directory
        if self.data_directory[-1] != "/":
            self.data_directory += "/"
    def load(self,file):
        data = []
        r = open(self.data_directory+file+".txt","r")
        for line in r:
            data.append([int(line[0]),line[2:-1]])
        return(data)
        
### Exemple de fonctionnement :
# Loader = DataLoader("C:/Users/Antoine/Documents/GitHub/TransferLearningProject/data")
# data = Loader.load("instruments")


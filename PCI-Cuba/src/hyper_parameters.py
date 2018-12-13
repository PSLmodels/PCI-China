import pickle 

class hyper_parameters:
    def __init__(self, varirate, fixed):
        self.varirate = varirate
        self.fixed = fixed
        self.perf = dict() 

    def to_dictionary(self):
        my_dict = dict()
        my_dict.update(self.varirate)
        my_dict.update(self.fixed)
        my_dict.update(self.perf)

        return my_dict

    def save(self,folder,file=None):
        if file==None:
            file = self.fixed['mod_id'] + '.pkl'
        with open(folder +file , 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file):
        with open(file, 'rb') as f:
            x = pickle.load(f)
        return x
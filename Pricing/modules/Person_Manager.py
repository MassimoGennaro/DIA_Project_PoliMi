import random

##### PERSON MANAGER #####
# contiene informazioni sulle persone, categorie e altro.

class Person_Manager():
    def __init__(self, categories, probabilities,features):
        self.categories = categories
        self.n_categories = len(self.categories)
        self.probabilities = probabilities
        self.features = features
        # self.bound_num_persons = bound_num_persons

        self.persons_count = 0
        self.categories_count = [0]*self.n_categories
        # restituisce la categoria di una nuova persona, ogni persona è identificata completamente dalla categoria.

    def new_person(self):
        p_category = random.randint(0, self.n_categories-1) # [0,1,2]
        self.persons_count += 1
        self.categories_count[p_category] += 1
        return p_category



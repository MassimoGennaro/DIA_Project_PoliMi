#modules
import numpy as np
import random


from .learner import *
from .utils_functions import *






##### ENVIRONMENT #####
# contiene informazioni sui candidati e probabilità.
# le probabilità di ogni categoria possono variare se variano le fasi

# quando t avanza di tempo, può cambiare fase.
class Personalized_Environment():
    def __init__(self, arms_candidates, probabilities):
        # arms_candidates è array dei valori di ogni arm (e.g. [5, 10 ,15, 20 ,25])
        self.arms_candidates = arms_candidates
        # probabilities è un tensore in 3 dimensioni: [phase][category][arm]
        self.probabilities = probabilities
        self.time = 0


    # rende la reward del candidato in base alla [phase][category][arm]
    def round(self, p_category, pulled_arm):

        p = self.probabilities[p_category][pulled_arm]
        reward = np.random.binomial(1,p)
        self.time += 1
        return reward * self.arms_candidates[pulled_arm]


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

    # def change_phase(self, probabilities):
    #     self.probabilities = probabilities

# ogni contesto si occupa di un sottoinsieme dello spazio delle features
# se la persona appartiene al suo contesto, se ne occupa il suo learner
# un contesto insieme di tuple, ciascuna un elemento dello spazio delle features
class Context():
    def __init__(self, context_id, subspace, learner):
        # ogni contesto viene identificato progressivamente da un id
        self.context_id = context_id

        # variables è il sottospazio delle feature di cui si occupa il contesto
        # è una lista di tuple lunghe quanto il numero di variabili. (e.g. {(y,f), (y,u)})
        self.subspace = subspace

        # ad ogni contesto è associato un relativo learner
        self.learner = learner
        self.num_variables = 2
        self.rewards_log = []

    def update(self, features_person, pulled_arm, reward):
        # Aggiorna i parametri beta del learner
        self.learner.update(pulled_arm, reward)
        #registro una lista delle reward, con essa devo effettuare la split
        self.rewards_log.append((features_person, pulled_arm, reward))

#dal log toglie tutti i dati che riguardano una feature che NON vogliamo considerare
    def fetch_log(self, feature):
        new_log = []
        for i in range(len(self.rewards_log)):
            if feature not in self.rewards_log[i][0]:
                new_log.append(self.rewards_log[i])
        return new_log

#dato un log calcola il learner derivante da quel log
    def learner_sub_context(self, log, candidates_values):
        new_learner = TS_Learner_candidate(self.learner.n_arms)
        for i in range(len(log)):
            new_learner.update(log[i][1], log[i][2])
        return new_learner

#data una feature dalla quale splittare calcola i valori di expected reward POST split. Sono la parte sinistra della split condition
    def val_after_split(self, feature, candidates_values):
        sub_1 = self.fetch_log(feature)
        sub_2 = [x for x in self.rewards_log if x not in sub_1]
        # TODO : FIX THIS #
        if len(self.rewards_log) > 0:
            prob_1 = len(sub_1)/len(self.rewards_log)
            prob_2 = len(sub_2)/len(self.rewards_log)
        else:
            prob_1 = 0
            prob_2 = 0
        ###################
        lern_1 = self.learner_sub_context(sub_1, candidates_values)
        lern_2 = self.learner_sub_context(sub_2, candidates_values)
        exp_1 = lern_1.best_arm_lower_bound(candidates_values)
        exp_2 = lern_2.best_arm_lower_bound(candidates_values)
        ris = [prob_1*exp_1+prob_2*exp_2, lern_1, lern_2]
        return ris

#verifica che la split condition sia verificata e nel caso restituisce una tupla con (feature, val_after_split, lerner  sub contesto 1, lerner  sub contesto 2)
    def split_condition(self, feature, candidates_values):
        ris = []
        val_after_split = self.val_after_split(feature, candidates_values)
        if val_after_split[0] > self.learner.best_arm_lower_bound(candidates_values):
            ris = [feature, val_after_split[0], val_after_split[1], val_after_split[2]]

        return ris


    def split(self, candidates_values):
        # considero il learner e calcolo il valore atteso del best arm

        # Per ogni variabile, calcolo il valore atteso del best arm per i possibili valori
        # scelgo l'indice del best arm, [[[quello selezionato di più]]]--> NO!
        #best_arm = np.argmax(np.sum(self.learner.beta_parameters, axis=1)) SBAGLIATO!!
        #best arm è quello con miglior prodotto probabilità x value
        best_arm = self.learner.best_arm(candidates_values)
        # calcolo il suo valore atteso, deve essere usato per la split condition
        #best_expected_value = best_expected_value(self.learner.beta_parameters, best_arm, candidates_values)

        #split per i quali la split condition è soddisfatta, poi dovrò scecgliere il miglior candidato
        candidate_split = []
        # lista dei diversi valori di ogni variabile (1 o 2 se binaria)
        # devo scegliere indice della var
        count_var_values = [[] for x in range(self.num_variables)]
        for t in self.subspace:

            for var in range(self.num_variables):

                if t[var] not in count_var_values[var]:
                    count_var_values[var].append(t[var])


        # ottengo quanti valori diversi della variabile si occupa il contesto
        # scelgo solo le variabili del contesto con almeno due valori diversi, con 1 o 0
        splittable_vars = [1 if len(x)>=2 else 0 for x in count_var_values]


        for index, var in enumerate(splittable_vars):
            if var == 0:
                pass
            else:
                tmp_split_condition = self.split_condition(count_var_values[index][0], candidates_values)
                if len(tmp_split_condition) > 0:
                    candidate_split.append(tmp_split_condition)
        #restituisco una tupla che ha al primo posto lo spazio delle feature, al secodno il valore della split condition, terzo e quarto i lerner associati

        if len(candidate_split) > 0:
            return candidate_split[np.argmax([a[1] for a in candidate_split])]
        else:
            return candidate_split

# Context_Manager si occupa della gestione dei context-learner
# feature_space = [("y", "f"), ("y", "u"),("a", "f"),("a", "u")]
# TODO: si deve usare le week per decidere quando splittare
class Context_Manager():
    def __init__(self, n_arms, feature_space, categories, candidates, week = -1, contexts_known = False):
        # feature space è lista di tuple di dimensione pari al numero di variabili
        # feature space è l'unione di tutti i possibili contesti
        self.n_arms = n_arms
        self.feature_space = feature_space
        # ogni assegnamento dello spazio è gestito da un contesto
        if contexts_known == True:
            self.features_context = {categories[i]:i for i in range(len(categories))}
            #{("y", "f"):0, ("a", "f"):1,  ("y", "u"):2}
            self.contexts_set = {}
        else:
            self.features_context = {self.feature_space[i]:0 for i in range(len(feature_space))}
            #{("y", "f"):0, ("y", "u"):0, ("a", "f"):0, ("a", "u"):0 }
            self.contexts_set = {0:Context(0, feature_space, TS_Learner_candidate(n_arms))}

        # week se diverso da -1 effettua split ogni week (e.g. week=5 il giorno 4 splitta)
        self.week = week
        self.time = 0


    def add_context(self, subspace):
        new_id = len(self.contexts_set)
        self.contexts_set[new_id] = Context(new_id, subspace, TS_Learner_candidate(self.n_arms))
        for t in subspace:
            self.features_context[t] = new_id

    def select_arm(self, person_category, time, candidates_values):
        self.split(time, candidates_values)

        context_id = self.features_context[person_category]
        selected_arm = self.contexts_set[context_id].learner.pull_arm(candidates_values)
        return selected_arm

    def update_context(self, features_person, pulled_arm, reward):
        # ottengo l'id del contesto dalle feature della persona
        context_id = self.features_context[features_person]
        # chiamo update del learner del contesto
        self.contexts_set[context_id].update(features_person, pulled_arm, reward)

    # split alla fine di una week, considera ogni contesto per vedere se effettuare lo split
    def split(self, time, candidates_values):
        if (self.week != -1) and ((time+1)%self.week == 0):
            # TODO: EFFETTUA SPLIT PER OGNI CONTESTO

            contexts_set_copy = self.contexts_set.copy()

            for index, context in self.contexts_set.items():
                split = context.split(candidates_values)

                #se lo split non restituisce una stringa vuota significa che bisogna effettuarlo
                if split != []:
                    #viene eliminato il contesto padre e inseriti due nuovi contesti, che sono complementari nello spazio delle feature tra di loro rispetto al padre
                    number = len(contexts_set_copy.items())

                    compl_feature_1 = [x for x in context.subspace if split[0] not in x]
                    compl_feature_2 = [x for x in context.subspace if split[0] in x]



                    #viene aggiunto il primo sub contesto, con un nuovo numero, le sue feature e il suo lerner
                    contexts_set_copy[number] = Context(number, compl_feature_1, split[3])

                    #viene aggiunto il secondo sub contesto, con il numero del padre, le sue feature e il suo lerner
                    contexts_set_copy[index] = Context(index, compl_feature_2, split[2])

            self.contexts_set = contexts_set_copy
            # TODO: per ogni contesto, controlla il subspace
            #####
            # per ogni tupla nel subspace del contesto, aggiorna self.features_context con l'indice del contesto
            for context in self.contexts_set.values():
                for tup in context.subspace:
                    for key in self.features_context.keys():
                        if tup == key:
                            self.features_context[key] = context.context_id
            ####
            #print(time+1)
            #print(self.features_context)
        else:
            pass



# questa classe si occupa di creare i contesti, i learner e le persone.
class General():
    def __init__(self, p_manager, c_manager, environment):
        # assegno a General la classe Person_Manager
        self.person_manager = p_manager
        self.context_manager = c_manager
        self.environment = environment
        self.rewards_log = []
        self.candidates_values = self.environment.arms_candidates
        self.expected_values = [[c*0.5 for c in self.candidates_values] for cat in range(3)]

    # general effettua una simulazione, restituisce rewards_log
    def play_experiment(self, num_persons):
        for t in range(num_persons):
            # una persona è identificata dalla propria categoria
            category_person = self.person_manager.new_person()
            # ottengo le features della persona in base a category
            features_person = self.person_manager.categories[category_person]
            # c_manager sceglie l'arm usando learner associato alle feature
            candidates_values = self.environment.arms_candidates
            pulled_arm = self.context_manager.select_arm(features_person, t, candidates_values)
            # data category e arm, ottengo reward positiva o nulla
            reward_person = self.environment.round(category_person, pulled_arm)

            self.context_manager.update_context(features_person, pulled_arm, reward_person)

            # aggiorno il log dell'experiment, con esso posso fare context split
            # [[category, arm,reward]*]
            self.rewards_log.append([category_person, pulled_arm, reward_person])

        return self.rewards_log

    def run_pricing_experiment(self, n_categories_clicks):
        for index, clicks in enumerate(n_categories_clicks):
            features_person = self.person_manager.categories[index]

            for n in range(int(round(clicks))):
                pulled_arm = self.context_manager.select_arm(features_person, n, self.candidates_values)
                reward_person = self.environment.round(index, pulled_arm)
                self.context_manager.update_context(features_person, pulled_arm, reward_person)
                self.rewards_log.append([index, pulled_arm, reward_person])

            idx = self.context_manager.features_context[features_person]

            for c in range(len(self.candidates_values)):
                self.expected_values[idx][c] = self.context_manager.contexts_set[idx].learner.expected_value(c, self.candidates_values[c])
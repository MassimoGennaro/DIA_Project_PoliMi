#modules
import numpy as np
import random


#from .learner import *
from learner import *



def complementary_feature(feature):
    features_set = (("y", "a"),("f", "u"))
    for variable in features_set:
        if feature in variable:
            for f in variable:
                if f != feature:
                    return f


##### ENVIRONMENT #####
# contiene informazioni sui candidati e probabilità.
# le probabilità di ogni categoria possono variare se variano le fasi

# quando t avanza di tempo, può cambiare fase.
class Personalized_Environment():
    def __init__(self, arms_candidates, probabilities, horizon):
        # arms_candidates è array dei valori di ogni arm (e.g. [5, 10 ,15, 20 ,25])
        self.arms_candidates = arms_candidates
        # probabilities è un tensore in 3 dimensioni: [phase][category][arm]
        self.probabilities = probabilities
        # lista di intervalli delle fasi: e.g [(1,100), (101,200), (201, 300)]
        #self.phases = phases
        self.horizon = horizon
        self.time = 0
        # parte dalla prima fase
        #self.current_phase = 0

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
        self.probabilities = probabilities
        self.features = features
        # self.bound_num_persons = bound_num_persons

        self.persons_count = 0
        self.categories_count = [0, 0, 0]
        # restituisce la categoria di una nuova persona, ogni persona è identificata completamente dalla categoria.

    def new_person(self):
        p_category = random.randint(0,2)
        self.persons_count += 1
        self.categories_count[p_category] += 1
        return p_category

    # def change_phase(self, probabilities):
    #     self.probabilities = probabilities

# ogni contesto si occupa di un sottoinsieme dello spazio delle features
# se la persona appartiene al suo contesto, se ne occupa il suo learner
# un contesto insieme di tuple, ciascuna un elemento dello spazio delle features
class Context():
    def __init__(self, context_id, subspace, learner, logs=[]):
        # ogni contesto viene identificato progressivamente da un id
        self.context_id = context_id

        # variables è il sottospazio delle feature di cui si occupa il contesto
        # è una lista di tuple lunghe quanto il numero di variabili. (e.g. {(y,f), (y,u)})
        self.subspace = subspace

        # ad ogni contesto è associato un relativo learner
        self.learner = learner
        self.num_variables = 2
        self.rewards_log = logs
        #self.rewards_log = []


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
        # lista valori possibili in questo contesto di ogni variabile (1 o 2)
        count_var_values = [[] for x in range(self.num_variables)]
        # per ogni elemento del subspace del contesto
        for t in self.subspace:
            # per ogni variabile 
            for var in range(self.num_variables):
                if t[var] not in count_var_values[var]:
                    count_var_values[var].append(t[var])
        
        # scelgo solo variabili con almeno due valori diversi, con mask di 0 e 1
        splittable_vars = [1 if len(x)>=2 else 0 for x in count_var_values]

        # per ogni variabile del contesto splittabile, valuto il valore prima e dopo split
        for index, var in enumerate(splittable_vars):
            # se non splittabile, non la considero
            if var == 0:
                pass
            else:
                # considero la split condition della variabile
                tmp_split_condition = self.split_condition(count_var_values[index][0], candidates_values)
                if len(tmp_split_condition) > 0:
                    candidate_split.append(tmp_split_condition)
        #restituisco una tupla :(subspace-feature, valore di split condition, learner contesto 0 , learner contesto 1)
        # subspace-feature è valore della variabile per lo split, i due learner sono associati ai due nuovi contesti
        
        # scelgo la split che massimizza valore di split condition
        if len(candidate_split) > 0:
            return candidate_split[np.argmax([a[1] for a in candidate_split])]
        # se vuota, restituisco lista vuota
        else:
            return candidate_split


    # verifica split condition secondo tale valore di una variabile, restituisce
    # (feature, val_after_split, learner  sub contesto 1, learner  sub contesto 2)
    def split_condition(self, feature, candidates_values):
        ris = []
        # calcolo valore dopo lo split secondo tale feature
        val_after_split = self.val_after_split(feature, candidates_values)
        # se supera il lower bound del best arm
        # 
        if val_after_split[0] > self.learner.best_arm_lower_bound(candidates_values):
            ris = [feature, val_after_split[0], val_after_split[1], val_after_split[2]]
            
        return ris


    # Data feature da splittare, calcola i valori di expected reward POST split.
    # è a sinistra nella split condition
    def val_after_split(self, feature, candidates_values):
        # divido il log secondo tale feature
        # sub-1 è log dei dati con feature negata
        sub_1 = self.fetch_log(feature)
        # sub-1 è log dei dati con feature positiva
        #sub_2 = [x for x in self.rewards_log if x not in sub_1]
        sub_2 = self.fetch_log(complementary_feature(feature))

        # devo trovare il value after split
        # calcolo probabilità dei due diversi valori della variabile
        if len(self.rewards_log) > 0:
            prob_1 = len(sub_1)/len(self.rewards_log)
            prob_2 = len(sub_2)/len(self.rewards_log)
        else: 
            prob_1 = 0
            prob_2 = 0
        
        # NEW! salvo il t del learner attuale del contesto, dovrò aggiornare il t del nuovo learner
        actual_t = self.learner.t

        # creo due nuovi learner associati ai due diversi contesti possibili
        # addestro tali learner con l'esperienza accumulata, in base al loro contesto
        # learn 1 considera log SENZA feature
        learn_1 = self.learner_sub_context(sub_1, candidates_values)
        # learn 1 considera log CON feature
        learn_2 = self.learner_sub_context(sub_2, candidates_values)

        # NEW!
        # aggiorno il t dei nuovi learner
        learn_1.t = actual_t
        learn_2.t = actual_t

        # calcolo lower bound dei due learner
        exp_1 = learn_1.best_arm_lower_bound(candidates_values)
        exp_2 = learn_2.best_arm_lower_bound(candidates_values)
        ris = [prob_1*exp_1+prob_2*exp_2, learn_1, learn_2]
        return ris


	# dal log toglie i dati senza feature che NON vogliamo considerare
    def fetch_log(self, feature):
        new_log = []
        for i in range(len(self.rewards_log)):
            if feature not in self.rewards_log[i][0]:
                new_log.append(self.rewards_log[i])
        return new_log

	#dato un log, crea un learner addestrato con l'esperienza del log
    def learner_sub_context(self, log, candidates_values):
    	# creo nuovo learner
    	# TODO: i nuovi learner devono possedere lo stesso t, per essere più sicuri
    	# inoltre i nuovi learner devono aggiornarsi con i vecchi log.
        new_learner = TS_Learner_candidate(self.learner.n_arms)
        # faccio update del learner
        # NEW! qui aggiorno con l'attuale log, ma non con il vecchio!
        # i nuovi learner non apprendono dal passato log, solo dal presente log

        for i in range(len(log)):
            new_learner.update(log[i][1], log[i][2])
            

        return new_learner

    
    def update(self, features_person, pulled_arm, reward):
        # Aggiorna i parametri beta del learner
        self.learner.update(pulled_arm, reward)
        #registro una lista delle reward, con essa devo effettuare la split
        self.rewards_log.append((features_person, pulled_arm, reward))




# Context_Manager si occupa della gestione dei context-learner
# feature_space = [("y", "f"), ("y", "u"),("a", "f"),("a", "u")]
# TODO: si deve usare le week per decidere quando splittare
class Context_Manager():
    def __init__(self, n_arms, feature_space, candidates, week = -1):
        # feature space è lista di tuple di dimensione pari al numero di variabili
        # feature space è l'unione di tutti i possibili contesti
        self.feature_space = feature_space
        # ogni assegnamento dello spazio è gestito da un contesto
        self.features_context = {("y", "f"):0, ("y", "u"):0, ("a", "f"):0, ("a", "u"):0 }
        # ogni contesto ha id, subspace feature e learner. Era un dizionario, inutile?
        self.contexts_set = {0:Context(0, feature_space, TS_Learner_candidate(n_arms))}
        #self.contexts_set = {Context(0, feature_space, TS_Learner_candidate(n_arms))}
        # week se diverso da -1 effettua split ogni week (e.g. week=5 il giorno 4 splitta)
        self.week = week
        # time va aggiornato ad ogni nuova persona
        self.time = 0

    def select_arm(self, person_category, time, candidates_values):
        self.time = time
        self.split(self.time, candidates_values)

        context_id = self.features_context[person_category]
        #print(context_id) ####################à
        selected_arm = self.contexts_set[context_id].learner.pull_arm(candidates_values)
        #print(self.contexts_set[context_id].subspace) ####################
        #print() ####################
        return selected_arm


    # split alla fine di una week, considera ogni contesto per vedere se effettuare lo split
    def split(self, time, candidates_values):
    	# effettuo split se week!=1 ed è t corrisponde
        if (self.week != -1) and ((time+1)%self.week == 0):
            # copio insieme dei contesti attuale            
            contexts_set_copy = self.contexts_set.copy()
            # ciclo su ogni contesto del contexts_set
            for index, context in self.contexts_set.items():
            	# chiamo context.split(), re
                split = context.split(candidates_values)
                #se lo split non restituisce una stringa vuota significa che bisogna effettuarlo
                # cosa possiede split?
                if split != []:
                    # trovo nuovo indice cel nuovo contesto

                    feature = split[0]
                    # val_after_split = split[1]
                    
                    # LEARNER AGGIORNATI
                    learner_1 = split[2] # questo learner è associato al log SENZA la feature
                    learner_2 = split[3] # questo learner è associato al log CON la feature

                    # LEARNER NUOVI
                    #learner_1 = TS_Learner_candidate(context.learner.n_arms)
                    #learner_2 = TS_Learner_candidate(context.learner.n_arms)

                    # LEARNER COPIA DEL PADRE
                    #learner_1 = TS_Learner_candidate(context.learner.n_arms)
                    #learner_2 = TS_Learner_candidate(context.learner.n_arms)

                    print("beta parameters learner 0")
                    print(context.learner.beta_parameters)
                    print("beta parameters learner 1")
                    print(learner_1.beta_parameters)
                    print("beta parameters learner 2")
                    print(learner_2.beta_parameters)

                    number = len(contexts_set_copy.items())
                    # split[2] e split[3] sono associati a feature in che modo?
                    # compl_feature_1 sottospazio SENZA la feature
                    compl_feature_1 = [x for x in context.subspace if feature not in x]
                    # compl_feature_1 sottospazio CON la feature
                    compl_feature_2 = [x for x in context.subspace if feature in x]
                    
                    # nuovo sotto-contesto, nuovo numero, subspace SENZA feature, suo learner
                    #contexts_set_copy[number] = Context(number, compl_feature_1, split[3])
                    # aggiorno contesto padre con il numero del padre, subspace CON feature, suo learner
                    #contexts_set_copy[index] = Context(index, compl_feature_2, split[2])
###############################

					# TODO CAMBIA context.rewards_log con contesto 1 e contesto 2
                    # NEW! i nuovi contesti ora hanno i learner associati correttamente con i contesti#
                    # nuovo sotto-contesto, nuovo numero, subspace SENZA feature, suo learner
                    log_1 = context.fetch_log(feature)
                    log_2 = context.fetch_log(complementary_feature(context.fetch_log(feature)))

                    contexts_set_copy[number] = Context(number, compl_feature_1, learner_1, log_1) #context.rewards_log
                    # aggiorno contesto padre con il numero del padre, subspace CON feature, suo learner
                    contexts_set_copy[index] = Context(index, compl_feature_2, learner_2, log_2)
###############################
            
            # aggiorno il context_set con quello nuovo dopo aver creato i nuovi contesti 
            self.contexts_set = contexts_set_copy

            # aggiorna self.features_context con l'indice del contesto corretto
            for context in self.contexts_set.values():
            	for tup in context.subspace:
            		for key in self.features_context.keys():
            			if tup == key:
            				self.features_context[key] = context.context_id
            				
            # Elenca nuovi contesti
            for c in self.contexts_set.values():
            	print(c.context_id, c.subspace)
            else:
            	print("\n")
            # condizione non rispettata, non effettuo split
        else:
            pass


    def update_context(self, features_person, pulled_arm, reward):
        # ottengo l'id del contesto dalle feature della persona
        context_id = self.features_context[features_person]
        # chiamo update del learner del contesto
        self.contexts_set[context_id].update(features_person, pulled_arm, reward)


# questa classe si occupa di creare i contesti, i learner e le persone.
class General():
    def __init__(self, p_manager, c_manager, environment):
        # assegno a General la classe Person_Manager
        self.person_manager = p_manager
        self.context_manager = c_manager
        self.environment = environment
        self.rewards_log = []


    # general effettua una simulazione, restituisce rewards_log
    def play_experiment(self, num_persons):
        for t in range(num_persons):
            # nuova persona
            category_person = self.person_manager.new_person()
            features_person = self.person_manager.categories[category_person]
            #print(features_person) ####################
            # candidates
            candidates_values = self.environment.arms_candidates
            # scelgo arm
            pulled_arm = self.context_manager.select_arm(features_person, t, candidates_values)
            # ottengo reward positiva o nulla
            reward_person = self.environment.round(category_person, pulled_arm)

            # aggiorno context_manager con reward
            self.context_manager.update_context(features_person, pulled_arm, reward_person)

            # aggiorno il log dell'experiment, con esso analizzo regret
            self.rewards_log.append([category_person, pulled_arm, reward_person])


        return self.rewards_log

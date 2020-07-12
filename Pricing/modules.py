#modules
import numpy as np
import random



#from .learner import *
from Pricing.learner import *
#from learner import *


#from .learner import *






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
        print("Creato context_id {}".format(context_id))
        print("len(rewards_log) = {}".format(len(self.rewards_log)))
        print("subspace = {}".format(self.subspace))
        print("beta_parameters \n{}".format(self.learner.beta_parameters.astype(int)))
        
        
    def update(self, features_person, pulled_arm, reward):
        # Aggiorna i parametri beta del learner
        self.learner.update(pulled_arm, reward)
        #registro una lista delle reward, con essa devo effettuare la split
        new_log = (features_person, pulled_arm, reward)
        self.rewards_log.append(new_log)
        #print("contesto {} update n* {}".format(self.context_id, len(self.rewards_log)-1))



#dal log toglie tutti i dati che riguardano una feature che NON vogliamo considerare
    def fetch_log(self, feature):
        #print(feature, "feature")
        new_log = []
        for i in range(len(self.rewards_log)):
            if feature not in self.rewards_log[i][0]:
                new_log.append(self.rewards_log[i])
        #print(len(self.rewards_log), len(new_log), "log, fetch log senza feature")
        return new_log

#dato un log calcola il learner derivante da quel log
    def learner_sub_context(self, log, candidates_values):
        new_learner = TS_Learner_candidate(self.learner.n_arms)
        for i in range(len(log)):
            new_learner.update(log[i][1], log[i][2])
        return new_learner


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
        # lista valori possibili in questo contesto di ogni variabile (1 o 2)
        count_var_values = [[] for x in range(self.num_variables)]
        # per ogni elemento del subspace del contesto
        for t in self.subspace:

            # per ogni variabile 
            for var in range(self.num_variables):
                if t[var] not in count_var_values[var]:
                    count_var_values[var].append(t[var])
        
        # scelgo solo variabili con almeno due valori diversi, con mask di 0 e 1

        # ottengo quanti valori diversi della variabile si occupa il contesto
        # scelgo solo le variabili del contesto con almeno due valori diversi, con 1 o 0

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

        #restituisco una tupla che ha al primo posto lo spazio delle feature, al secodno il valore della split condition, terzo e quarto i lerner associati

        if len(candidate_split) > 0:
            return candidate_split[np.argmax([a[1] for a in candidate_split])]
        # se vuota, restituisco lista vuota
        else:
            return candidate_split


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
        print("Context Manager creato")
        if contexts_known == True:
            self.features_context = {categories[i]:i for i in range(len(categories))}
            #{("y", "f"):0, ("a", "f"):1,  ("y", "u"):2}
            self.contexts_set = {}
        else:
            self.features_context = {self.feature_space[i]:0 for i in range(len(feature_space))}
            #{("y", "f"):0, ("y", "u"):0, ("a", "f"):0, ("a", "u"):0 }
            # Istanzio il nuovo contesto 0, con id 0 e rewards_log iniziale vuoto
            rewards_log_start = []
            context_start = Context(0, feature_space, TS_Learner_candidate(n_arms), rewards_log_start)
            self.contexts_set = {0:context_start}

        # week se diverso da -1 effettua split ogni week (e.g. week=5 il giorno 4 splitta)
        self.week = week
        # time va aggiornato ad ogni nuova persona
        self.time = 0


    def add_context(self, subspace):
        new_id = len(self.contexts_set)
        self.contexts_set[new_id] = Context(new_id, subspace, TS_Learner_candidate(self.n_arms))
        for t in subspace:
            self.features_context[t] = new_id

    def select_arm(self, person_category, time, candidates_values):
        self.time = time
        self.split(self.time, candidates_values)

        context_id = self.features_context[person_category]
        
        selected_arm = self.contexts_set[context_id].learner.pull_arm(candidates_values)


        #print(context_id) 
        #print(self.contexts_set[context_id].subspace) 
        #print() 
        return selected_arm


    # split alla fine di una week, considera ogni contesto per vedere se effettuare lo split
    def split(self, time, candidates_values):
    	# effettuo split se week!=1 ed è t corrisponde
        if (self.week != -1) and ((time+1)%self.week == 0):
            print(time)

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

                    #print("beta parameters learner 0")
                    #print(context.learner.beta_parameters)
                    #print("beta parameters learner 1")
                    #print(learner_1.beta_parameters)
                    #print("beta parameters learner 2")
                    #print(learner_2.beta_parameters)

                    number = len(contexts_set_copy.items())
                    # split[2] e split[3] sono associati a feature in che modo?
                    # compl_feature_1 sottospazio SENZA la feature
                    compl_feature_1 = [x for x in context.subspace if feature not in x]
                    # compl_feature_1 sottospazio CON la feature
                    compl_feature_2 = [x for x in context.subspace if feature in x]

                    #print(feature, compl_feature_1, compl_feature_2)
                    
                    # nuovo sotto-contesto, nuovo numero, subspace SENZA feature, suo learner
                    #contexts_set_copy[number] = Context(number, compl_feature_1, split[3])
                    # aggiorno contesto padre con il numero del padre, subspace CON feature, suo learner
                    #contexts_set_copy[index] = Context(index, compl_feature_2, split[2])
###############################

                    # TODO CAMBIA context.rewards_log con contesto 1 e contesto 2
                    # NEW! i nuovi contesti ora hanno i learner associati correttamente con i contesti#
                    # nuovo sotto-contesto, nuovo numero, subspace SENZA feature, suo learner
                    log_1 = context.fetch_log(feature)
                    #log_2 = context.fetch_log(complementary_feature(context.fetch_log(feature)))
                    log_2 = context.fetch_log(complementary_feature(feature))
                    #print(feature, complementary_feature(feature))
                    #print(len(log_1), "log 1")
                    #print(len(log_2), "log 2")
                    #print(len(context.rewards_log), "log tot")

                    contexts_set_copy[number] = Context(number, compl_feature_1, learner_1, log_1) #context.rewards_log
                    # aggiorno contesto padre con il numero del padre, subspace CON feature, suo learner
                    contexts_set_copy[index] = Context(index, compl_feature_2, learner_2, log_2)
###############################
            
            # aggiorno il context_set con quello nuovo dopo aver creato i nuovi contesti 

                    #viene eliminato il contesto padre e inseriti due nuovi contesti, che sono complementari nello spazio delle feature tra di loro rispetto al padre
                    #number = len(contexts_set_copy.items())

            
            self.contexts_set = contexts_set_copy
            for i, c in self.contexts_set.items():
                print("context {}".format(i))
                print("len(context.rewards_log) = {}".format(len(c.rewards_log)))
                print("context.subspace = {}".format(c.subspace))
                print("context.learner.beta_parameters = \n{}".format(c.learner.beta_parameters.astype(int)))

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
            # TODO: perchè si ripete questo doppio for?

                for tup in context.subspace:
                    for key in self.features_context.keys():
                        if tup == key:
                            self.features_context[key] = context.context_id
            ####
            #print(time+1)
            #print(self.features_context)

        else:
            pass


    def update_context(self, features_person, pulled_arm, reward):
        # ottengo l'id del contesto dalle feature della persona
        context_id = self.features_context[features_person]
        # chiamo update del learner del contesto
        self.contexts_set[context_id].update(features_person, pulled_arm, reward)

    def val_att_arm(self, features_person, pulled_arm, candidates_values):
    	# ottengo l'id del contesto dalle feature della persona
        context_id = self.features_context[features_person]
        alpha = self.contexts_set[context_id].learner.beta_parameters[pulled_arm][0]
        beta = self.contexts_set[context_id].learner.beta_parameters[pulled_arm][1]
        price = candidates_values[pulled_arm]

        probab_arm = alpha/(alpha + beta)
        expected_value = probab_arm * price

        return expected_value


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
            # nuova persona
            category_person = self.person_manager.new_person()
            features_person = self.person_manager.categories[category_person]
            #print("persona {}, category {}".format(t, features_person))
            
            # candidates
            candidates_values = self.environment.arms_candidates
            # scelgo arm
            pulled_arm = self.context_manager.select_arm(features_person, t, candidates_values)

            #NEW!: ritorna valore atteso di tale arm dal contesto assegnato
            valore_atteso_arm = self.context_manager.val_att_arm(features_person, pulled_arm, candidates_values)

            # ottengo reward positiva o nulla
            reward_person = self.environment.round(category_person, pulled_arm)

            # aggiorno context_manager con reward
            self.context_manager.update_context(features_person, pulled_arm, reward_person)

            # aggiorno il log dell'experiment, con esso analizzo regret
            #NEW! 
            self.rewards_log.append([category_person, pulled_arm, reward_person, valore_atteso_arm])
            #self.rewards_log.append([category_person, pulled_arm, reward_person])
        
        

        for i, c in self.context_manager.contexts_set.items():
            print("\ncontext {}".format(i))
            print("len(context.rewards_log) = {}".format(len(c.rewards_log)))
            print("context.subspace = {}".format(c.subspace))
            print("context.learner.beta_parameters = \n{}".format(c.learner.beta_parameters.astype(int)))
            
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
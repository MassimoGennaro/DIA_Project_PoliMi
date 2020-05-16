#modules
import numpy as np
import random

from dia_code import *
from learner import *
from utils_functions import *





##### ENVIRONMENT #####
# contiene informazioni sui candidati e probabilità.
# le probabilità di ogni categoria possono variare se variano le fasi

# quando t avanza di tempo, può cambiare fase.
class Personalized_Environment():
    def __init__(self, arms_candidates, probabilities, phases, horizon):
        # arms_candidates è array dei valori di ogni arm (e.g. [5, 10 ,15, 20 ,25])
        self.arms_candidates = arms_candidates
        # probabilities è un tensore in 3 dimensioni: [phase][category][arm]
        self.probabilities = probabilities
        # lista di intervalli delle fasi: e.g [(1,100), (101,200), (201, 300)]
        self.phases = phases
        self.horizon = horizon
        self.time = 0
        # parte dalla prima fase
        self.current_phase = 0

    # rende la reward del candidato in base alla [phase][category][arm]
    def round(self, p_category, pulled_arm):
        # se time è oltre fase attuale, aumento di 1
        if time > self.phases[self.current_phase][1]:
            self.current_phase += 1
        p = self.probabilities[self.current_phase][p_category][pulled_arm]
        reward = np.random.binomial(1,p)
        self.time += 1
        return reward * self.arms_candidates[pulled_arm]


##### PERSON MANAGER #####
# contiene informazioni sulle persone, categorie e altro.

class Person_Manager():
    def __init__(self, categories, probabilities, bound_num_persons, features):
        self.categories = categories
        self.probabilities = probabilities
        self.features = features
        self.bound_num_persons = bound_num_persons

        self.persons_count = 0
        self.categories_count = [0, 0, 0]
        # restituisce la categoria di una nuova persona, ogni persona è identificata completamente dalla categoria.

    def new_person(self):
        p_category = random.randint(0,2)
        self.persons_count += 1
        self.categories_count[p_category] += 1
        return p_category

    def change_phase(self, probabilities):
        self.probabilities = probabilities

# ogni contesto si occupa di un sottoinsieme dello spazio delle features
# se la persona appartiene al suo contesto, se ne occupa il suo learner
# un contesto insieme di tuple, ciascuna un elemento dello spazio delle features
class Context():
    def __init__(self, context_id, subspace, learner):
        # ogni contesto viene identificato progressivamente da un id
        #self.context_id = context_id inutile?

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
    def fetch_log(feature):
        new_log = []
        for i in range(len(self.rewards_log)) if feature not in self.rewards_log[i][0]:
            new_log.append(self.rewards_log[i])

#dato un log calgola il learner derivante da quel log
    def learner_sub_context(log, candidates_values):
        new_learner = TS_Learner(self.learner.n_arms)
        for i in range(len(log)):
            new_learner.update(log[i][1], log[i][2])
        return new_lerner

#data una feature dalla quale splittare calcola i valori di expected reward POST split. Sono la parte sinistra della split condition
    def val_after_split(feature, candidates_values):
        sub_1 = fetch_log(feature)
        sub_2 = [x in self.rewards_log not in sub_1]
        prob_1 = len(sub_1)/len(rewards_log)
        prob_2 = len(sub_2)/len(rewards_log)
        lern_1 = learner_sub_context(sub_1, candidates_values)
        lern_2 = learner_sub_context(sub_2, candidates_values)
        exp_1 = lern_1.best_arm_lower_bound(candidates_values)
        exp_2 = lern_2.best_arm_lower_bound(candidates_values)
        ris = [prob_1*exp_1+prob_2*exp_2, lern_1, lern_2]
        return ris

#verifica che la split condition sia verificata e nel caso restituisce una tupla con (feature, val_after_split, lerner  sub contesto 1, lerner  sub contesto 2)
    def split_condition(feature, candidates_values):
        ris = []
        val_after_split = val_after_split(feature, candidates_values)
        if val_after[0] > self.learner.best_arm_lower_bound(candidates_values):
            ris = [feature, val_after[0], val_after[1], val_after[2]]
        return ris


    def split(self, candidates_values):
        # considero il learner e calcolo il valore atteso del best arm

        # Per ogni variabile, calcolo il valore atteso del best arm per i possibili valori
        # scelgo l'indice del best arm, [[[quello selezionato di più]]]--> NO!
        #best_arm = np.argmax(np.sum(self.learner.beta_parameters, axis=1)) SBAGLIATO!!
        #best arm è quello con miglior prodotto probabilità x value
        best_arm = self.learner.best_arm(candidate_value)
        # calcolo il suo valore atteso, deve essere usato per la split condition
        best_expected_value = best_expected_value(self.learner.beta_parameters, best_arm, candidates_values)

        #split per i quali la split condition è soddisfatta, poi dovrò scecgliere il miglior candidato
        candidate_split = []
        # lista dei diversi valori di ogni variabile (1 o 2 se binaria)
        # devo scegliere indice della var
        #TODO verificacre che le 5 linee seguenti facciano quello che dovrebbero fare
        count_var_values = [[] for x in range(self.num_variables)]
        for t in subspace:
            for var in range(self.num_variables):
                if t[var] not in count_var_values[var]:
                    count_var_values[var].append(t[var])
        #fine linee da controllare

        # ottengo quanti valori diversi della variabile si occupa il contesto
        # scelgo solo le variabili del contesto con almeno due valori diversi, con 1 o 0
        splittable_vars = [1 if len(x)>=2 else 0 for x in count_var_values]


        for var in splittable_vars:
            if var == 0:
                pass
            else:
                #TODO: count_var_values[var](qualcosa) da set A TUPLE, dovrebbe essere risolto, se le linee da controllare (app 136-140 sono corrette)
                candidadate_split.add(split_condition(count_var_values[var][0], candidates_values))
        #restituisco una tupla che ha al primo posto lo spazio delle feature, al secodno il valore della split condition, terzo e quarto i lerner associati
        return argmax(candidate_split, axis=1)

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
        #self.contexts_set = {0:Context(0, feature_space, TS_Learner_candidate(n_arms))}
        self.contexts_set = {Context(0, feature_space, TS_Learner_candidate(n_arms))}
        # week se diverso da -1 effettua split ogni week (e.g. week=5 il giorno 4 splitta)
        self.week = week
        self.time = 0

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
        if (time+1)%week == 0:
            # TODO: EFFETTUA SPLIT PER OGNI CONTESTO
            for index, context in self.contexts_set.items():
                split = context.split(candidates_values)
                #se lo split non restituisce una stringa vuota significa che bisogna effettuarlo
                if split != []:
                    #viene eliminato il contesto padre e inseriti due nuovi contesti, che sono complementari nello spazio delle feature tra di loro rispetto al padre
                    number = self.contexts_set.items()
                    compl_feature = [x in context.feature_space not in split[0]]
                    #viene aggiunto il primo sub contesto, con un nuovo numero, le sue feature e il suo lerner
                    contexts_set.add(Context(number, compl_feature, split[3]))
                    #viene aggiunto il secondo sub contesto, con il numero del padre, le sue feature e il suo lerner
                    context_set.add(Context(index, split[0], split[2]))
                    context_set.delete(context)

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

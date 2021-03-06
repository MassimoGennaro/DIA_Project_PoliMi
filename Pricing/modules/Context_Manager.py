from Pricing.learners.learner import TS_Learner_candidate
import numpy as np

def complementary_feature(feature):
    '''
    this function serves to find the opposite value of a variable, given a value
    '''
    features_set = (("y", "a"),("f", "u"))
    for variable in features_set:
        if feature in variable:
            for f in variable:
                if f != feature:
                    return f



class Context():
    """
    a context is assigned to a sub-set of the feature space.
    each user of that sub-set is managed by the learner of the corresponding context
    """

    def __init__(self, context_id, subspace, learner, logs=[], print_init=True):
        # context id
        self.context_id = context_id

        # sub-set of the feature spce
        # set of tuples of values, for each variable (e.g. {(y,f), (y,u)})
        self.subspace = subspace

        # each context contains a learner
        self.learner = learner
        self.num_variables = 2

        self.rewards_log = logs
        # if (print_init):
        #     print("\n")
        #     print("Create context: {}".format(context_id))
        #     print("Features in the context: {}".format(self.subspace))
        #     print("Beta parameters initialization:\n{}".format(self.learner.beta_parameters.astype(int)))

    def update(self, features_person, pulled_arm, reward):
        '''
        updates the learner's beta parameters
        '''
        self.learner.update(pulled_arm, reward)
        # update the rewards log
        new_log = (features_person, pulled_arm, reward)
        self.rewards_log.append(new_log)
        # print("contesto {} update n* {}".format(self.context_id, len(self.rewards_log)-1))

    
    def fetch_log(self, feature):
        '''
        returns the list of the rewards of the users that don't have that feature.
        used for the context split
        '''
        new_log = []
        for i in range(len(self.rewards_log)):
            if feature not in self.rewards_log[i][0]:
                new_log.append(self.rewards_log[i])
        # print(len(self.rewards_log), len(new_log), "log, fetch log senza feature")
        return new_log

    
    def learner_sub_context(self, log, candidates_values):
        '''
        given a log of rewards, returns a learner trained with it
        '''
        new_learner = TS_Learner_candidate(self.learner.n_arms)
        for i in range(len(log)):
            new_learner.update(log[i][1], log[i][2])
        return new_learner

    
    def split_condition(self, feature, candidates_values):
        '''
        calculate the split condition, if it's verified returns
        (feature, val_after_split, lerner  sub contesto 1, lerner  sub contesto 2)

        '''
        ris = []
        val_after_split = self.val_after_split(feature, candidates_values)
        if val_after_split[0] > self.learner.best_arm_lower_bound(candidates_values):
            ris = [feature, val_after_split[0], val_after_split[1], val_after_split[2]]

        return ris

    def split(self, candidates_values):
        '''
        checks if the context can be split
        '''
        # considero il learner e calcolo il valore atteso del best arm

        # Per ogni variabile, calcolo il valore atteso del best arm per i possibili valori
        best_arm = self.learner.best_arm(candidates_values)
        # calcolo il suo valore atteso, deve essere usato per la split condition

        # split per i quali la split condition è soddisfatta, poi dovrò scecgliere il miglior candidato
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

        splittable_vars = [1 if len(x) >= 2 else 0 for x in count_var_values]

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

        # restituisco una tupla :(subspace-feature, valore di split condition, learner contesto 0 , learner contesto 1)
        # subspace-feature è valore della variabile per lo split, i due learner sono associati ai due nuovi contesti

        # scelgo la split che massimizza valore di split condition

        # restituisco una tupla che ha al primo posto lo spazio delle feature, al secodno il valore della split condition, terzo e quarto i lerner associati

        if len(candidate_split) > 0:
            return candidate_split[np.argmax([a[1] for a in candidate_split])]
        # se vuota, restituisco lista vuota
        else:
            return candidate_split

    
    def val_after_split(self, feature, candidates_values):
        '''
        Calculates the expected reward POST split.
        '''

        # divido il log secondo tale feature
        # sub-1 è log dei dati con feature negata
        sub_1 = self.fetch_log(feature)

        # sub-1 è log dei dati con feature positiva
        sub_2 = self.fetch_log(complementary_feature(feature))

        # devo trovare il value after split
        # calcolo probabilità dei due diversi valori della variabile
        if len(self.rewards_log) > 0:
            prob_1 = len(sub_1) / len(self.rewards_log)
            prob_2 = len(sub_2) / len(self.rewards_log)
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

        # aggiorno il t dei nuovi learner
        learn_1.t = actual_t
        learn_2.t = actual_t

        # calcolo lower bound dei due learner
        exp_1 = learn_1.best_arm_lower_bound(candidates_values)
        exp_2 = learn_2.best_arm_lower_bound(candidates_values)
        ris = [prob_1 * exp_1 + prob_2 * exp_2, learn_1, learn_2]
        return ris




class Context_Manager():
    '''
    Manages the allocation and split of the contexts
    '''
    def __init__(self, n_arms, feature_space, categories, candidates, week=-1, contexts_known=False):
        # feature space è lista di tuple di dimensione pari al numero di variabili
        # feature space è l'unione di tutti i possibili contesti
        self.n_arms = n_arms
        self.feature_space = feature_space
        # ogni assegnamento dello spazio è gestito da un contesto
        # print("Context Manager creato")
        if contexts_known == True:
            self.features_context = {categories[i]: i for i in range(len(categories))}
            # {("y", "f"):0, ("a", "f"):1,  ("y", "u"):2}
            self.contexts_set = {}
        else:
            self.features_context = {self.feature_space[i]: 0 for i in range(len(feature_space))}
            # {("y", "f"):0, ("y", "u"):0, ("a", "f"):0, ("a", "u"):0 }
            # Istanzio il nuovo contesto 0, con id 0 e rewards_log iniziale vuoto
            rewards_log_start = []
            context_start = Context(0, feature_space, TS_Learner_candidate(n_arms), rewards_log_start)
            self.contexts_set = {0: context_start}

        # week se diverso da -1 effettua split ogni week (e.g. week=5 il giorno 4 splitta)
        self.week = week
        # time va aggiornato ad ogni nuova persona
        self.time = 0

        self.subspace_sequence = {}

    def add_context(self, subspace, print_init=True):
        '''
        manually adds a new context, with a given subspace
        '''
        new_id = len(self.contexts_set)
        self.contexts_set[new_id] = Context(new_id, subspace, TS_Learner_candidate(self.n_arms), print_init=print_init)
        for t in subspace:
            self.features_context[t] = new_id

    def select_arm(self, person_category, time, candidates_values):
        '''
        returns the arm's candidate to propose to the person
        '''
        self.time = time
        self.split(self.time, candidates_values)

        context_id = self.features_context[person_category]

        selected_arm = self.contexts_set[context_id].learner.pull_arm(candidates_values)

        # print(context_id)
        # print(self.contexts_set[context_id].subspace)
        # print()
        return selected_arm

    
    def split(self, time, candidates_values):
        '''
        at each week, checks the split condition of each context, for each different variable.
        if the context it's splittable, splits it with the best variable
        '''

        # effettuo split se week!=1 ed è t corrisponde
        if (self.week != -1) and ((time + 1) % self.week == 0):
            # print("----------------")
            # print("Splitting at: {}".format(time))

            # copio insieme dei contesti attuale
            contexts_set_copy = self.contexts_set.copy()
            # ciclo su ogni contesto del contexts_set

            for index, context in self.contexts_set.items():
                # chiamo context.split(), re
                split = context.split(candidates_values)

                # se lo split non restituisce una stringa vuota significa che bisogna effettuarlo
                if split != []:
                    # trovo nuovo indice cel nuovo contesto

                    feature = split[0]
                    learner_1 = split[2]  # questo learner è associato al log SENZA la feature
                    learner_2 = split[3]  # questo learner è associato al log CON la feature
                    # print("beta parameters learner 0")
                    # print(context.learner.beta_parameters)
                    # print("beta parameters learner 1")
                    # print(learner_1.beta_parameters)
                    # print("beta parameters learner 2")
                    # print(learner_2.beta_parameters)

                    number = len(contexts_set_copy.items())
                    # split[2] e split[3] sono associati a feature in che modo?
                    # compl_feature_1 sottospazio SENZA la feature
                    compl_feature_1 = [x for x in context.subspace if feature not in x]
                    # compl_feature_1 sottospazio CON la feature
                    compl_feature_2 = [x for x in context.subspace if feature in x]

                    # print(feature, compl_feature_1, compl_feature_2)

                    # nuovo sotto-contesto, nuovo numero, subspace SENZA feature, suo learner
                    log_1 = context.fetch_log(feature)
                    log_2 = context.fetch_log(complementary_feature(feature))
                    # print(feature, complementary_feature(feature))
                    # print(len(log_1), "log 1")
                    # print(len(log_2), "log 2")
                    # print(len(context.rewards_log), "log tot")

                    contexts_set_copy[number] = Context(number, compl_feature_1, learner_1,
                                                        log_1)  # context.rewards_log
                    # aggiorno contesto padre con il numero del padre, subspace CON feature, suo learner
                    contexts_set_copy[index] = Context(index, compl_feature_2, learner_2, log_2)

            # aggiorno il context_set con quello nuovo dopo aver creato i nuovi contesti

            # viene eliminato il contesto padre e inseriti due nuovi contesti, che sono complementari nello spazio delle feature tra di loro rispetto al padre
            # number = len(contexts_set_copy.items())

            self.contexts_set = contexts_set_copy
            """print("Contexts after splitting:")
            for i, c in self.contexts_set.items():

                print("Context {}".format(i))
                #print("len(context.rewards_log) = {}".format(len(c.rewards_log)))
                print("Features subspace in context {} = {}".format(i,c.subspace))
                print("Beta parameters of context {} = \n{}".format(i,c.learner.beta_parameters.astype(int)))"""

            # aggiorna self.features_context con l'indice del contesto corretto
            for context in self.contexts_set.values():

                for tup in context.subspace:
                    for key in self.features_context.keys():
                        if tup == key:
                            self.features_context[key] = context.context_id

            # Elenca nuovi contesti
            self.subspace_sequence[time] = {}
            for c in self.contexts_set.values():
                # print(c.context_id, c.subspace)
                for e in c.subspace:
                    self.subspace_sequence[time][e] = c.context_id
            else:
                # print("\n")
                # TODO: perchè si ripete questo doppio for?

                for tup in context.subspace:
                    for key in self.features_context.keys():
                        if tup == key:
                            self.features_context[key] = context.context_id
            ####
            # print(time+1)
            # print(self.features_context)

        else:
            pass

    def update_context(self, features_person, pulled_arm, reward):
        '''
        given the reward, arm and features, update the corresponding context
        '''
        # ottengo l'id del contesto dalle feature della persona
        context_id = self.features_context[features_person]
        # chiamo update del learner del contesto
        self.contexts_set[context_id].update(features_person, pulled_arm, reward)

    def val_att_arm(self, features_person, pulled_arm, candidates_values):
        '''
        returns the expected value of the arm of a given context
        '''
        # ottengo l'id del contesto dalle feature della persona
        context_id = self.features_context[features_person]
        alpha = self.contexts_set[context_id].learner.beta_parameters[pulled_arm][0]
        beta = self.contexts_set[context_id].learner.beta_parameters[pulled_arm][1]
        price = candidates_values[pulled_arm]

        probab_arm = alpha / (alpha + beta)
        expected_value = probab_arm * price

        return expected_value

# questa classe si occupa di creare i contesti, i learner e le persone.
class General():
    def __init__(self, p_manager, c_manager, environment):
        # assegno a General la classe Person_Manager
        self.person_manager = p_manager
        self.context_manager = c_manager
        self.environment = environment
        self.rewards_log = []
        self.candidates_values = self.environment.arms_candidates
        self.expected_values = [[c * 0.5 for c in self.candidates_values] for cat in range(3)]

    # general effettua una simulazione, restituisce rewards_log
    def play_experiment(self, num_persons):
        for t in range(num_persons):
            # nuova persona
            category_person = self.person_manager.new_person()
            features_person = self.person_manager.categories[category_person]
            # print("persona {}, category {}".format(t, features_person))

            # candidates
            candidates_values = self.environment.arms_candidates
            # scelgo arm
            pulled_arm = self.context_manager.select_arm(features_person, t, candidates_values)

            # ritorna valore atteso di tale arm dal contesto assegnato
            valore_atteso_arm = self.context_manager.val_att_arm(features_person, pulled_arm, candidates_values)

            # ottengo reward positiva o nulla
            reward_person = self.environment.round(category_person, pulled_arm)

            # aggiorno context_manager con reward
            self.context_manager.update_context(features_person, pulled_arm, reward_person)

            # aggiorno il log dell'experiment, con esso analizzo regret
            self.rewards_log.append([category_person, pulled_arm, reward_person, valore_atteso_arm])
            # self.rewards_log.append([category_person, pulled_arm, reward_person])

        # for i, c in self.context_manager.contexts_set.items():
        #     print("\ncontext {}".format(i))
        #     print("len(context.rewards_log) = {}".format(len(c.rewards_log)))
        #     print("context.subspace = {}".format(c.subspace))
        #     print("context.learner.beta_parameters = \n{}".format(c.learner.beta_parameters.astype(int)))

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
                self.expected_values[idx][c] = self.context_manager.contexts_set[idx].learner.expected_value(c,
                                                                                                             self.candidates_values[
                                                                                                                 c])

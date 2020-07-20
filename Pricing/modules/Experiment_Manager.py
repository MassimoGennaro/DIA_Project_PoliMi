
class General():
    '''
    this class runs the experiments with given number of users of each category
    '''
    def __init__(self, p_manager, c_manager, environment):
        self.person_manager = p_manager
        self.context_manager = c_manager
        self.environment = environment
        self.rewards_log = []
        self.candidates_values = self.environment.arms_candidates
        self.expected_values = [[c * 0.5 for c in self.candidates_values] for cat in range(3)]

    
    def play_experiment(self, num_persons):
        '''
        runs a pricing experiment with a given number of users
        '''
        for t in range(num_persons):
            # nuova persona
            category_person = self.person_manager.new_person()
            features_person = self.person_manager.categories[category_person]
            # print("persona {}, category {}".format(t, features_person))

            # candidates
            candidates_values = self.environment.arms_candidates
            # scelgo arm
            pulled_arm = self.context_manager.select_arm(features_person, t, candidates_values)

            # returns the expected value of this arm from the assigned context.
            valore_atteso_arm = self.context_manager.val_att_arm(features_person, pulled_arm, candidates_values)

            # receive positive or null reward
            reward_person = self.environment.round(category_person, pulled_arm)

            # update context_manager with reward
            self.context_manager.update_context(features_person, pulled_arm, reward_person)

            # update ethe experiment's rewads log,
            self.rewards_log.append([category_person, pulled_arm, reward_person, valore_atteso_arm])
            

        # for i, c in self.context_manager.contexts_set.items():
        #     print("\ncontext {}".format(i))
        #     print("len(context.rewards_log) = {}".format(len(c.rewards_log)))
        #     print("context.subspace = {}".format(c.subspace))
        #     print("context.learner.beta_parameters = \n{}".format(c.learner.beta_parameters.astype(int)))

        return self.rewards_log

    def run_pricing_experiment(self, n_categories_clicks):
        '''
        runs a pricing experiment with a given number of users.
        this is called as a method inside the advertising modules
        '''
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

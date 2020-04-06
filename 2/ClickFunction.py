class ClickFunction():
    def function(self, x, user, phase):
        if user == 'FaY' and phase == 'M':
            return (1-np.exp(-x)) * 10
        if user == 'FaY' and phase == 'E':
            return (1-np.exp(-x)) * 15
        if user == 'FaY' and phase == 'W':
            return (1-np.exp(-x)) * 30
        if user == 'FaA' and phase == 'M':
            return (1-np.exp(-x)) * 1
        if user == 'FaA' and phase == 'E':
            return (1-np.exp(-x)) * 10
        if user == 'FaA' and phase == 'W':
            return (1-np.exp(-x)) * 15
        if user == 'NFaY' and phase == 'M':
            return (1-np.exp(-x)) * 7
        if user == 'NFaY' and phase == 'E':
            return (1-np.exp(-x)) * 5
        if user == 'NFaY' and phase == 'W':
            return (1-np.exp(-x)) * 10

    def aggregate_function(self, x, user):
        return (5*(self.function(x, user, 'M')+self.function(x, user, 'E'))+2*self.function(x, user, 'W'))/7

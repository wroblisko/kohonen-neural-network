def distance1D(self, n1, n2):
    pos1, pos2 = self.neuron_position[n1], self.neuron_position[n2]
    if pos1==pos2:
        return 1
    if abs(pos1-pos2)==1:
        return 0.5
    else:
        return 0

def empty_distance(self, n1, n2):
    pos1, pos2 = self.neuron_position[n1], self.neuron_position[n2]
    if pos1==pos2:
        return 1
    else:
        return 0

def distance2D(self, n1, n2):
    pos1, pos2 = self.neuron_position[n1], self.neuron_position[n2] 
    if pos1==pos2:
        return 1
    elif abs(pos1-pos2)==1:
        return 0.5
    elif abs(pos1-pos2)==self.neurons_in_row:
        return 0.5
    else:
        return 0

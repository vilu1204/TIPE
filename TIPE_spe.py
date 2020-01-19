##def 

import math
import random 
import tkinter 
import pickle 
import numpy as np


epsilon = 10**-2
eta = 10**-2
mut_rate = 0.3

def error(choice,sol,test=False): 
    x1 = choice[0] / sqrt(choice[0]**2 + choice[1]**2)
    y1 = choice[1] / sqrt(choice[0]**2 + choice[1]**2)
    x2 = sol[0] / sqrt(sol[0]**2 + sol[1]**2)
    y2 = sol[1] / sqrt(sol[0]**2 + sol[1]**2)
    if test :
        print(x1,y1,"/",x2,y2)
    return sqrt((x1-x2)**2 + (y1-y2)**2)
    


def activation_function(x):
    return np.arctan(x)/(pi/2)




class fish :
    
    def __init__(self, Id, batch_ref, species, preys, predators, NN, x, y, speed, view, size):
        self.Id = Id # identifiant du poisson, facilite le traitement
        self.species = species # espece, rang dans l'échelle alimentaire
        self.preys = preys # liste de booleens donnant les especes de proies 
        self.predators = predators #liste de booleens donnant les especes predatrices
        self.NN = NN # réseau de neurones
        self.x = x
        self.y = y
        self.batch_ref = batch_ref #batch du poisson
        self.speed = speed #vitesse
        self.view = view # distance a laquelle le poisson voit 
        self.size = size # taille du poisson sur l'affichage graphique - rayon du cercle qui le représente / portee a laquelle le poisson peut manger les autres
        self.alive = True
        self.eaten = 0
        self.death_turn = 0
        self.score = 0
    
    
    def kill(self): # tue le poisson 
        self.alive = False
        self.death_turn = self.batch_ref.turn
    
    
    
    def fish_search(self,d0): #renvoie la liste des poisson proches a moins de d0
        sight = []
        for f in self.batch_ref.list_fishes:
            if f.alive and abs(self.x - f.x) + abs(self.y - f.y) <= d0:
                sight.append(f)
        return sight
    
    
    def eat_around(self): #tue / mange les proies a portee 
        reach = self.fish_search(2*self.size)
        for f in reach :
            if self.preys[f.species] :
                f.kill()
                self.eaten +=1
    
    
    def move(self, Dx, Dy):
        if self.alive :
            self.eat_around() # on appelle la fonction avant le mouvement au cas où une proie s'est approchée dans son mouvement
            if Dx != 0.0 and Dy != 0.0:
                x0 = self.x + self.speed* Dx 
                y0 = self.y + self.speed* Dy 
                if x0 < 0:
                    x0 = 0
                elif x0 > self.batch_ref.dim: 
                    x0 = self.batch_ref.dim
                if y0 < 0:
                    y0 = 0
                elif y0 > self.batch_ref.dim: 
                    y0 = self.batch_ref.dim
                self.x = x0
                self.y = y0
                self.eat_around()

    
    
    def randmove(self):
        self.eat_around() # on appelle la fonction avant le mouvement au cas où une proie s'est approchée dans son mouvement
        self.move(random()-.5,random()-.5)
    
    
    def gather_input_data(self):
        x0, y0 = self.x, self.y
        dist = self.batch_ref.dim
        res = [x0/dist, (dist - x0)/dist, y0/dist, (dist - y0)/dist]
        sight = self.fish_search(self.view)
        for f in self.batch_ref.list_fishes :
            if f.species != self.species:
                res.append((f.x-x0)/dist) 
                res.append((f.y-y0)/dist)
                res.append((f.x-x0 + f.y-y0)/dist) #position relative et distance du poisson considere
        res.append(.1) #biais
        return np.array(res)
    
    
    def decide(self,data):
        res = data
        for layer in self.NN:
            res[-1] = .1 #biais
            res = activation_function(res.dot(layer))
        return res[0], res[1]

    
    def choose_move(self): #choix du mouvement a effectuer par le reseau
        self.eat_around() # on appelle la fonction avant le mouvement au cas où une proie s'est approchée dans son mouvement
        tmp = self.decide(self.gather_input_data())
        self.move(tmp[0], tmp[1])
        
    
    def train(self, prey, pred):
        for situation in prey:
            for f in self.batch_ref.list_fishes :
                if self.preys[f.species]:
                    data = np.array([0] * self.batch_ref.NN_format[0])
                    data[-1] = 1
                    if f.species> self.species:
                        diff = self.batch_ref.nb_by_species[self.species]
                    else :
                        diff = 0
                    k = (f.Id - diff)*3 + 4 
                    data[k] = situation[0][0]
                    data[k+1] = situation[0][1]
                    data[k+2] = situation[0][2]
                    for layer in self.NN:
                        for i in layer:
                            for j in range(len(i)):
                                i[j] += epsilon 
                                Err1 = error(self.decide(data), situation[1])
                                i[j] -= 2*epsilon 
                                Err2 = error(self.decide(data), situation[1])
                                i[j] += epsilon
                                dErr = (Err1 - Err2)/(2*epsilon)
                                i[j] -= eta*dErr
        for situation in pred:
            for f in self.batch_ref.list_fishes :
                if self.predators[f.species]:
                    data = np.array([0] * self.batch_ref.NN_format[0])
                    data[-1] = 1
                    if f.species> self.species:
                        diff = self.batch_ref.nb_by_species[self.species]
                    else :
                        diff = 0
                    k = (f.Id - diff)*3 + 4 
                    data[k] = situation[0][0]
                    data[k+1] = situation[0][1]
                    data[k+2] = situation[0][2]
                    for layer in self.NN:
                        for i in layer:
                            for j in range(len(i)):
                                i[j] += epsilon 
                                Err1 = error(self.decide(data), situation[1])
                                i[j] -= 2*epsilon 
                                Err2 = error(self.decide(data), situation[1])
                                i[j] += epsilon
                                dErr = (Err1 - Err2)/(2*epsilon)
                                i[j] -= eta*dErr
    
    
    def avg_error(self, prey, pred):
        res = 0
        for situation in prey:
            for f in self.batch_ref.list_fishes :
                if self.preys[f.species]:
                    data = np.array([0] * self.batch_ref.NN_format[0])
                    data[-1] = 1
                    if f.species> self.species:
                        diff = self.batch_ref.nb_by_species[self.species]
                    else :
                        diff = 0
                    k = (f.Id - diff)*3 + 4 
                    data[k] = situation[0][0]
                    data[k+1] = situation[0][1]
                    data[k+2] = situation[0][2]
                    res += error(self.decide(data), situation[1])
        for situation in pred:
            for f in self.batch_ref.list_fishes :
                if self.predators[f.species]:
                    data = np.array([0] * self.batch_ref.NN_format[0])
                    data[-1] = 1
                    if f.species> self.species:
                        diff = self.batch_ref.nb_by_species[self.species]
                    else :
                        diff = 0
                    k = (f.Id - diff)*3 + 4 
                    data[k] = situation[0][0]
                    data[k+1] = situation[0][1]
                    data[k+2] = situation[0][2]
                    res += error(self.decide(data), situation[1])
        return res
    
    def session(self, prey, pred, n):
        for i in range (n):
            self.train(prey,pred)
        print(self.Id, self.avg_error(prey, pred))
    
    
    def mutant(self,Id):
        x = randint(self.batch_ref.dim*self.species//self.batch_ref.nb_species,self.batch_ref.dim*(self.species+1)//self.batch_ref.nb_species-1)        
        y = randint(self.batch_ref.dim*self.species//self.batch_ref.nb_species,self.batch_ref.dim*(self.species+1)//self.batch_ref.nb_species-1)
        res = [np.array([i[:] for i in layer]) for layer in self.NN]
        for layer in res:
            for i in layer :
                for j in range(len(i)):
                    i[j] *= 1 + mut_rate*(2*random()-1)
        return fish(Id, self.batch_ref, self.species, self.preys, self.predators, res, x, y, self.speed, self.view, self.size)



class batch :
    
    def __init__(self, nb_by_species, speeds, views, sizes, dim, preying, NN_format, turns_by_gen, children):
        self.nb_species = len (nb_by_species)  # nombre d'especes
        self.nb_by_species = nb_by_species # liste du nombre de poissons par espece
        self.nb = 0 # nombre total de poissons
        for i in nb_by_species :
            self.nb += i
        self.speeds = speeds # vitesse de chaque espece
        self.views = views # distance de vue de chaque espece
        self.sizes = sizes # taille de chaque espece
        self.dim = dim # taille du terrain 
        # self.field = [[[] for i in range(dim+1)] for j in range(dim+1)] # terrain , liste des poissons sur la case par ID, on a dim+1 cases pour contenir l'intervalle [0,dim]
        self.preying = preying # matrice de booleens determinant si une espece peut en manger une autre
        self.turn = 0
        self.NN_format = NN_format
        self.children = children
        self.gen = 0
        self.turns_by_gen = turns_by_gen
    
    
    def create(self):
        self.list_fishes = [0]*self.nb
        Id=0
        for i in range(self.nb_species) :
            for j in range (self.nb_by_species[i]):
                x , y = randint(i*self.dim//self.nb_species,(i+1)*self.dim//self.nb_species-1) , randint(i*self.dim//self.nb_species,(i+1)*self.dim//self.nb_species-1) #initialisation des coordonnees
                preys = self.preying[i][:]
                predators = [self.preying[k][i] for k in range(self.nb_species)]
                Network = [np.array([[2*random()-1 for i in range(self.NN_format[k+1])] for j in range(self.NN_format[k])]) for k in range(len(self.NN_format)-1)]
                self.list_fishes[Id] = fish(Id, self, i, preys, predators, Network, x, y, self.speeds[i], self.views[i], self.sizes[i])
                # self.field[floor(x)][floor(y)].append(Id)
                Id += 1

    def get(self,i):
        return self.list_fishes[i]
    
    def run_turn(self):
        self.turn +=1
        for i in range(self.nb):
            if self.get(i).alive:
                self.get(i).choose_move()
                
    
    def randturn(self):
        self.turn +=1
        for i in range(self.nb):
            if self.get(i).alive:
                self.get(i).randmove()
    
    
    def fullup(self, prey, pred, n):
        for f in self.list_fishes:
            f.session(prey, pred, n)

    
    
    def cplt_gen(self):
        while self.turn < self.turns_by_gen:
            self.run_turn()
    
    
    def rating(self):
        for f in self.list_fishes :
            if f.alive :
                f.score = (f.eaten+1)*self.turn*2
            else :
                f.score = (f.eaten+1)*f.death_turn
    
    
    def top(self):
        a = 0
        b = 0
        self.rating()
        res = [0]*self.nb_species
        for k in range(self.nb_species):
            n = self.nb_by_species[k] // self.children[k]
            a = b 
            b += self.nb_by_species[k] # bornes de la portion du tableau qui contient l'espece numero k
            select = [self.list_fishes[a+i] for i in range(n)] # on identifie les n meilleurs de l'espece k
            for i in range (n):
                for j in range(n-i-1):
                    if select[j].score < select[j+1].score:
                        select[j+1],select[j] = select[j],select[j+1]
            # on garde la liste des selectionnes triee, le moins bon est a la premiere place
            for i in range (n+a, b):
                if self.list_fishes[i].score > select[1].score:
                    select[1] = self.list_fishes[i]
                    for j in range (n-1): #comme seul le premier element est potentiellement mal range, un seul passage suffit
                        if select[j].score < select[j+1].score:
                            select[j+1],select[j] = select[j],select[j+1]
            for i in select:
                res[k] = select[:]
        return res
    
    
    
    def new_gen(self):
        self.cplt_gen()
        sample = self.top()
        new = [0]*self.nb
        id = 0
        for i in range(len(sample)) :
            for j in range(self.children[i]) :
                for f in sample[i] :
                    new[id] = f.mutant(id)
                    id +=1
        self.list_fishes = new[:]
        self.turn = 0
        self.gen += 1
    
    
    

b1 = batch([10,10,10],[1,1,1],[20,20,20],[1,1,1],50, [[False,False,False],[True,False,False],[False,True,False]], [65, 11, 2], 50, [2,2,2])
b1.create()


##training 

prey1 = [((1,0,1), (1,0)),
          ((0,1,1), (0,1)), 
          ((1,1,1), (1,1))]

pred1 = [ ((1,0,1), (-1,0)),
          ((0,1,1), (0,-1)),
          ((1,1,1), (-1,-1))]





##
b1.fullup(prey1, pred1, 0)

# b1.list_fishes[0].session(prey1,pred1,100)
# b1.list_fishes[10].session(prey1,pred1,100)
# b1.list_fishes[20].session(prey1,pred1,300)

##

for i in range(1,10):
    # b1.list_fishes[i].NN = b1.list_fishes[0].NN[:]
    # b1.list_fishes[10+i].NN = b1.list_fishes[10].NN[:]
    b1.list_fishes[20+i].NN = b1.list_fishes[20].NN[:]




## module graphique
b = b1


size = 20
fenetre=Tk()
canvas = Canvas(fenetre, width=size*(b.dim+1), height=size*(b.dim+1), bg="blue")


bouton=Button(fenetre, text="quitter", command=fenetre.destroy)
bouton.pack()

def disp():
    canvas.delete("all")
    for f in b.list_fishes :
        if f.alive :
            x = f.x
            y = f.y
            if f.species == 0:
                canvas.create_oval(size*(x-f.size),size*(y-f.size),size*(x+f.size),size*(y+f.size),fill='white')
            if f.species == 1:
                canvas.create_oval(size*(x-f.size),size*(y-f.size),size*(x+f.size),size*(y+f.size),fill='grey')
            if f.species == 2:
                canvas.create_oval(size*(x-f.size),size*(y-f.size),size*(x+f.size),size*(y+f.size),fill='black')


def clavier(event):
    touche = event.keysym
    if touche == "a" :
        b.randturn()
        disp()
    elif touche == "e" :
        b.run_turn()
        disp()
    elif touche == "r":
        b.cplt_gen()
        disp()
    elif touche == "t":
        b.new_gen()
        disp()
    elif touche == "p":
        for i in range(1000):
            b.new_gen()
        print(b.gen, "ok")
        disp()
        


disp()
canvas.focus_set()
canvas.bind("<Key>", clavier)
canvas.pack()
fenetre.mainloop()





##


# pickle.dump((b1) ,open('/media/alefebvre/3713-4F08/charsave', 'wb'))
pickle.dump((b1) ,open('E:charsave_v1_half_grad', 'wb'))

##

# b1 = pickle.load(open('/media/alefebvre/3713-4F08/charsave', 'rb'))
b1 = pickle.load(open('E:charsave_v1_half_grad', 'rb'))



























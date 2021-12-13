import pygame as py
import neat
import time
import os
import random
from car import Car
from road import Road
from world import World
from NNdraw import NN
from config_variables import *
import pickle
py.font.init()


bg = py.Surface((WIN_WIDTH, WIN_HEIGHT))
bg.fill(GRAY)


def draw_win(cars, road, world):     #x and y are the coordinates of the best car
    road.draw(world)
    for car in cars:
        car.draw(world)

    world.bestNN.draw(world)

    py.display.update()
    world.win.blit(bg, (0,0))       #blit of the background immediately after the update so if I have draws before draw_win they are not covered by the background

def main(genomes = [], config = []):

    nets = []
    ge = []
    cars = []
    t = 0

    world = World(STARTING_POS, WIN_WIDTH, WIN_HEIGHT)
    world.win.blit(bg, (0,0))

    NNs = []

    for _,g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        cars.append(Car(0, 0, 0))
        g.fitness = 0
        ge.append(g)
        NNs.append(NN(config, g, (90, 210)))

    road = Road(world)
    clock = py.time.Clock()

    run = True
    while run:
        t += 1
        clock.tick(FPS)
        world.updateScore(0)

        for event in py.event.get():
            if event.type == py.QUIT:
                run = False
                py.quit()
                quit()

        (xb, yb) = (0,0)
        i = 0
        while(i < len(cars)):

            car = cars[i]

            input = car.getInputs(world, road)
            input.append(car.vel/MAX_VEL)
            car.commands = nets[i].activate(tuple(input))

            y_old = car.y
            (x, y) = car.move(road,t)

            if t>10 and (car.detectCollision(road) or y > world.getBestCarPos()[1] + BAD_GENOME_TRESHOLD or y>y_old or car.vel < 0.1): #the t is used to avoid eliminating machines in the first few frames (in the first frames getCollision () always returns true)
                run = False
                break
            else:
                world.updateScore(ge[i].fitness)
                world.bestNN = NNs[i]
                world.bestInputs = input
                world.bestCommands = car.commands
                i += 1

            if y < yb:
                (xb, yb) = (x, y)

        world.updateBestCarPos((xb, yb))
        road.update(world)
        draw_win(cars, road, world)


#NEAT function
def run(config_path):
    # Load requried NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Unpickle saved winner
    with open("sd_car1.pkl", "rb") as f:
        genome = pickle.load(f)
        f.close()

    # for i, input in enumerate(config.genome_config.input_keys):
    #     print(input)
    # print("--------------")
    # for out in config.genome_config.output_keys:
    #     print(out)

    # Convert loaded genome into required data structure
    genomes = [(1, genome)]
    #print(genome)

    # Call game with only the loaded genome
    main(genomes, config)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_file.txt")
    run(config_path)

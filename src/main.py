import numpy as np 
import random
import math
# Данные которые принимает игрок:
# 1. свое положение в координатах
# 2. положение соперника в координатах
# 3. свой предыдущий ход(не модет ходить в 1 направлении 2 раза подряд)
# 4. предыдущий ход соперника(не модет ходить в 1 направлении 2 раза подряд)
# 5. сколько сделал ходов до победы
# 1.2 - возможно за место первых двух координат можно просто вычислять расстояние до соперника,
# либо по обычной формуле для координат, либо как то с учетом количества поворотов,
# мб расстояние - кол-во шагов кратчайшего пути до соперника на текущей доске, однако это
# не дает информации в какой именно стороне противник 
# 6. - heal

def save_p(population):
  f = open('population.txt', "w")
  for i in population:
    f.write(str(i)+ '\n')
  f.close()

def init_board():
  board = np.zeros((6,6), dtype=int)
  board[1][1] = 1 #p1
  board[4][4] = -1 #p2
  return board

def in_pl(w=1): # w= 1 or -1? this gender
  p1 = np.random.random(9)
  p1[4] = w
  p1[5] = 0
  p1[6] = 0
  p1[7] = 0
  p1[8] = 0
  return p1

def init_players():
  #f = open('population.txt', "r")
  p1 = np.random.random(6)
  p2 = np.random.random(6)
  p1[5] = 0
  p2[5] = 0
  p1[4] = 1
  p2[4] = -1
  return p1, p2

def coord_check(board, w):
  coord = []
  if w>0:
    coord.append(np.where(board>0)[0][0])
    coord.append(np.where(board>0)[1][0])
  elif w<0:
    coord.append(np.where(board<0)[0][0])
    coord.append(np.where(board<0)[1][0])
  return coord

def coord_restruct(board, w, new_coord):
  try:
    x,y = coord_check(board, w)
    board[x][y] = 0
    x,y = new_coord
    board[x][y] = w
  except:
    print("Невозможно изменить координаты")
    return False
  return True

def sigmoid(x):
  return 1/(1 + np.exp(-x+1.5)) # 0-1

def finfunc(x): 
  if x > 0.75:
    x = 3
    return x
  elif x > 0.5 and x<=0.75:
    x = 2
    return x
  elif x > 0.25 and x<=0.5:
    x = 1
    return x
  elif x <=0.25:
    x = 0
    return x

def move_math(board, p, last_move_me, last_move_p): # p - player, w = 1 or -1
  coord_me = coord_check(board, p[4])
  coord_p = coord_check(board, -p[4])
  f =(p[2]*(last_move_me/3) +p[3] *(last_move_p/3))/(p[1]*math.sqrt(math.pow(coord_p[0] - coord_me[0],2) + math.pow(coord_p[1] - coord_me[1],2)) )
  # фитнесс функция 
  return finfunc(sigmoid(f)) # moove

def move_go(board, move, w, cpu=False):
  x,y = coord_check(board,w)
  if move == 0:
    y-=1
  elif move == 1:
    x-=1
  elif move == 2:
    y+=1
  elif move == 3:
    x+=1
  if cpu == False: coord_restruct(board, w,[x,y])
  else: return [x,y]

def check_move(board, w, move, last_move):
  if move == last_move:
    return 0 # game over
  x,y = coord_check(board, w)
  if x == 0 or x == 5 or y == 0 or y == 5:
    return 1 # skipping a move
  suuum = []
  for i in board:
    suuum.append(sum(i))
  if sum(suuum) !=0:
    return 2 # winning
  return 'True'
  
def display(board):
  a=1
  #print("_"*90)
  #print(board)

def cpu_move(board, p, last_move):
  coord_p = coord_check( board, -p[4])
  move_pattern = [0,1,2,3]
  try:
    move_pattern.remove(last_move)
  except Exception:
    #print(Exception)
    a=1

  dist = []
  for i in move_pattern:
    coord_me = move_go(board,i,-p[4],True)
    dist.append([i, math.sqrt(math.pow(coord_p[0] - coord_me[0],2) + math.pow(coord_p[1] - coord_me[1],2))] )
  
  dist.sort(key=lambda dist: dist[1],reverse=True)
  return  dist[0][0] # moove

def cpu_ga(p):
  board = init_board()
  last_move_p,last_move_cpu = 4,4
  move_p,move_cpu = None, None
  for o in range(50): # после 50 ходов в холостую идет ниичья
    display(board)
    move_p = move_math(board,p,last_move_p, last_move_cpu)
    copy_board = board.copy()
    move_go(board, move_p, p[4])
    flag = check_move(board,p[4], move_p, last_move_p)
    if flag !='True':
      if flag == 0:
        p[5]-=1 # отнимаем здоровья за ход в 1 направлении
        p[6]+=1
        return p 
      elif flag == 1:
        board = copy_board.copy()
        p[7]+=1
      elif flag == 2:
        p[8]+=1
        p[5]+=60
        print("player_win")
        return p
    p[5]+=2
    last_move_p = move_p
    move_cpu = cpu_move(board, p, last_move_cpu)
    display(board)
    move_go(board,move_cpu,-p[4])
    flag = check_move(board,-p[4], move_cpu, last_move_cpu)
    if flag !='True':
      if flag == 0:
        p[6]+=5
        print("cpu_lag_move_x2")
        return p 
      elif flag == 1: board = copy_board.copy()
      elif flag == 2:
        p[5]-=60
        print("cpu_win")
        return p
    last_move_cpu = move_cpu
  print("прошло 50 ходов")
  p[5]-=10
  return p

def score(p1,p2):
  board = init_board()
  last_move_p1,last_move_p2 = 5,5
  move_p1,move_p2,copy_board = None, None,None
  for o in range(50): # после 50 ходов в холостую идет ниичья
    move_p1 = move_math(board,p1,last_move_p1, last_move_p2)
    copy_board = board.copy()
    move_go(board,move_p1, p1[4])
    display(board)
    flag = check_move(board,p1[4], move_p1, last_move_p1)
    if flag !='True':
      if flag == 0:
        p1[5]-=2 # отнимаем здоровья за ход в 1 направлении
        p2[5]+=2
        print("Тех поражение")
        return p2, p1 # Тип техническое поражение
      elif flag == 1:
        board = copy_board.copy()
        print("Пропуск хода ")
      elif flag == 2:
        print("Выигрышь честным путем! ")
        p1+=10
        p2-=5
        return p1, p2
    p1[5]+=1
    last_move_p1 = move_p1
    copy_board = board.copy()
    move_p2 = move_math(board,p2,last_move_p2, last_move_p1) 
    move_go(board, move_p2,p2[4])
    display(board)
    flag = check_move(board,p2[4], move_p2, last_move_p2)
    if flag != 'True':
      #print("Какого хера тут происходит ",flag )
      if flag == 0:
        p2[5]-=2
        p1[5]+=2
        print("Тех. поражениие")
        return p1, p2 # Тип техническое поражение
      elif flag == 1:
        board = copy_board.copy()
        print("Пропуск хода ")
      elif flag == 2:
        p2+=10
        p1-=5
        print("Выигрышь честным путем! ")
        return p2, p1
    p2[5]+=1
    last_move_p2 = move_p2
    print(move_p1, move_p2)
  print("Ничья")
  return p1, p2

def cross(p1:list,p2:list):
  t = random.randint(1,3)
  p1[t:4]= p2[t:4].copy()
  return p1

def myt(p:list):
  t = random.randint(1,3)
  p[t] = abs(1-p[t])
  return p


def init_population64(cpu=False):
  population = []
  if cpu == False:
    for i in range(32):
      a,b = init_players()
      population.append(a)
      population.append(b)
    return population
  else:
    for i in range(64):
      population.append(in_pl(1))
    return population
  
def game_all_and_all64(population:list): # каждый играет 64 раза (с каждым)
  for i in range(64):
    for j in range(i+1, 62):
      population[i],population[j] = score(population[i],population[j])
  return population

def otbor_and_recombo(population:list):
  population.sort(key=lambda population: population[5],reverse=True)
  new_population = []
  mi = min(abs(population[0][5]), abs(population[63][5]))
  ma = max(abs(population[0][5]), abs(population[63][5]))
  chance = abs(population[0][5])
  for individ in population:
    if abs(individ[5])/chance >= random.randint(mi,ma)/ma:
      new_population.append(individ) # отбираем новую популяцию вероятностным методом
    
  count = 64 - len(new_population)
  for i in range(0,count):
    new_population.append(cross(population[i],population[i+1]))
  
  for i in new_population:
    i[5]-=10
    if random.random() <= 0.4:
      i = myt(i)
  return new_population # может быть на 1 или несколько ячеек больше 64

import time
if __name__ == "__main__":
  start_time = time.time()
  #population = init_population64()
  #for i in range(50):
  #  population = game_all_and_all64(population)
  #  population = otbor_and_recombo(population)
  #print(population)
  population = init_population64(cpu = True)
  pop_copy = population.copy()
  for o in range(1000):
    for i in range(len(population)):
      population[i] = cpu_ga(population[i])
    population = otbor_and_recombo(population)
  end_time = time.time()

  print(end_time - start_time, '\n'*4)

  for i in population:
    print([round(j,2) for j in i])
import pygame
import random
import heapq
import sys
import argparse
import csv  
import os   
import time
from abc import ABC, abstractmethod

# ==========================
# CLASSES DE PLAYER
# ==========================
class BasePlayer(ABC):
    """
    Classe base para o jogador (robô).
    Para criar uma nova estratégia de jogador, basta herdar dessa classe e implementar o método escolher_alvo.
    """
    def __init__(self, position):
        self.position = position  # Posição no grid [x, y]
        self.cargo = 0            # Número de pacotes atualmente carregados
        self.battery = 90        # Nível da bateria
        self.perfil_bateria = 'aventureiro' # 'conservador', 'balanceado' ou 'aventureiro'

    @abstractmethod
    def escolher_alvo(self, world):
        """
        Retorna o alvo (posição) que o jogador deseja ir.
        Recebe o objeto world para acesso a pacotes e metas.
        """
        pass

    def pode_visitar_e_recarregar(self, custo_ate_destino, custo_ate_recarregador):
            """
            Verifica se o robô tem bateria suficiente para ir até o destino e depois até o recarregador.
            Os custos já devem ser previamente calculados.
            """
            custo_total = custo_ate_destino + custo_ate_recarregador

            if self.perfil_bateria == "conservador":
                return self.battery >= custo_total
            elif self.perfil_bateria == "balanceado":
                return self.battery >= custo_total * 0.8
            elif self.perfil_bateria == "aventureiro":
                return True
            else:
                return self.battery >= custo_total  # padrão: conservador

class HybridClusterPlayer(BasePlayer):

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar(self, start, goal, world):
        maze = world.map
        size = world.maze_size
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        close_set = set()
        gscore = {tuple(start): 0}
        fscore = {tuple(start): self.heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[tuple(start)], tuple(start)))
        while oheap:
            current = heapq.heappop(oheap)[1]
            if list(current) == goal:
                return gscore[current]
            close_set.add(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                tentative_g = gscore[current] + 1
                if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                    if maze[neighbor[1]][neighbor[0]] == 1:
                        continue
                else:
                    continue
                if neighbor in close_set and tentative_g >= gscore.get(neighbor, 0):
                    continue
                if tentative_g < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                    gscore[neighbor] = tentative_g
                    fscore[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return float('inf')

    def dijkstra_multi_target_completo(self, start, alvos, world):
        maze = world.map
        size = world.maze_size
        visited = set()
        dist = {}
        heap = [(0, tuple(start))]
        alvos_set = set(map(tuple, alvos))

        while heap and alvos_set:
            current_dist, current = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)

            if current in alvos_set:
                dist[current] = current_dist
                alvos_set.remove(current)
                if not alvos_set:
                    break

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                    if maze[neighbor[1]][neighbor[0]] == 0:
                        heapq.heappush(heap, (current_dist + 1, neighbor))

        return dist

    def pontuar_objetivo_cluster(self, alvo, todos, world, raio=10):
        proximos = 0
        for outro in todos:
            if outro != alvo:
                d = self.heuristic(alvo, outro)
                if d <= raio:
                    proximos += 1
        return proximos

    def escolher_melhor_com_cluster_eficiente(self, origem, candidatos, world):
        distancias = self.dijkstra_multi_target_completo(origem, candidatos, world)
        melhor_score = float('-inf')
        melhor_alvo = None

        for alvo in candidatos:
            custo = distancias.get(tuple(alvo), float('inf'))
            if custo == float('inf'):
                continue
            cluster = self.pontuar_objetivo_cluster(alvo, candidatos, world)
            score = cluster * 3 - custo
            if score > melhor_score:
                melhor_score = score
                melhor_alvo = alvo
        return melhor_alvo

    def escolher_alvo(self, world):
        pos_atual = self.position
        packages = world.packages
        goals = world.goals
        pgs = packages + goals
        custo_retorno = self.astar(pos_atual, world.recharger, world)

        if self.battery <= custo_retorno:
            return world.recharger

        if self.cargo == 0 and packages:
            alvo = self.escolher_melhor_com_cluster_eficiente(pos_atual, packages, world)
            dist = self.astar(pos_atual, alvo, world)
            dist_ret = self.astar(alvo, world.recharger, world)
            return alvo if self.pode_visitar_e_recarregar(dist, dist_ret) else world.recharger

        if self.cargo == len(goals):
            alvo = self.escolher_melhor_com_cluster_eficiente(pos_atual, goals, world)
            dist = self.astar(pos_atual, alvo, world)
            dist_ret = self.astar(alvo, world.recharger, world)
            return alvo if self.pode_visitar_e_recarregar(dist, dist_ret) else world.recharger

        if self.cargo > 0 and packages:
            alvo = self.escolher_melhor_com_cluster_eficiente(pos_atual, pgs, world)
            dist = self.astar(pos_atual, alvo, world)
            dist_ret = self.astar(alvo, world.recharger, world)
            return alvo if self.pode_visitar_e_recarregar(dist, dist_ret) else world.recharger

        return None

class astarPlayer(BasePlayer):
    
    def heuristic2(self, a, b):
        # Distância de Manhattan
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar2(self, start, goal,world):
        maze = world.map  # Acesso ao mapa do mundo
        size = world.maze_size
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        close_set = set()
        gscore = {tuple(start): 0}
        fscore = {tuple(start): self.heuristic2(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[tuple(start)], tuple(start)))
        
        while oheap:
            current = heapq.heappop(oheap)[1]
            if list(current) == goal:
                return gscore[current]  # Retorna a distância total percorrida
            
            close_set.add(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                tentative_g = gscore[current] + 1
                if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                    if maze[neighbor[1]][neighbor[0]] == 1:
                        continue
                else:
                    continue
                if neighbor in close_set and tentative_g >= gscore.get(neighbor, 0):
                    continue
                if tentative_g < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                    gscore[neighbor] = tentative_g
                    fscore[neighbor] = tentative_g + self.heuristic2(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return float('inf')  # Retorna infinito se não houver caminho

    def escolher_alvo(self, world):
            pos_atual = self.position
            packages = world.packages
            goals = world.goals
            pgs = packages + goals
            custo_bateria = self.astar2(pos_atual, world.recharger, world)

            if self.battery <= custo_bateria:
                return world.recharger
            

            if self.cargo == 0 and packages:
                melhor = min(packages, key=lambda p: self.astar2(pos_atual, p, world))
                dist = self.astar2(pos_atual, melhor, world)
                dist_ret = self.astar2(melhor, world.recharger, world)
                return melhor if self.pode_visitar_e_recarregar(dist, dist_ret) else world.recharger

            if self.cargo == len(goals):
                melhor = min(goals, key=lambda g: self.astar2(pos_atual, g, world))
                dist = self.astar2(pos_atual, melhor, world)
                dist_ret = self.astar2(melhor, world.recharger, world)
                return melhor if self.pode_visitar_e_recarregar(dist, dist_ret) else world.recharger

            if self.cargo > 0 and packages:
                melhor = min(pgs, key=lambda x: self.astar2(pos_atual, x, world))
                dist = self.astar2(pos_atual, melhor, world)
                dist_ret = self.astar2(melhor, world.recharger, world)
                return melhor if self.pode_visitar_e_recarregar(dist, dist_ret) else world.recharger

            return None
           
class DijkstraPlayer(BasePlayer):

    def dijkstra_multi_target(self, start, targets, world):
        maze = world.map
        size = world.maze_size
        visited = set()
        dist = {tuple(start): 0}
        heap = [(0, tuple(start))]
        targets_set = set(map(tuple, targets))

        while heap:
            current_dist, current = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)

            if current in targets_set:
                return list(current), current_dist  # Alvo mais próximo encontrado

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                    if maze[neighbor[1]][neighbor[0]] == 0:
                        new_cost = current_dist + 1
                        if new_cost < dist.get(neighbor, float('inf')):
                            dist[neighbor] = new_cost
                            heapq.heappush(heap, (new_cost, neighbor))

        return None, float('inf')  

    def dijkstra(self, start, goal, world):
        return self.dijkstra_multi_target(start, [goal], world)[1]

    def escolher_alvo(self, world):
        pos_atual = self.position
        packages = world.packages
        goals = world.goals
        pgs = packages + goals
        custo_bateria = self.dijkstra(pos_atual, world.recharger, world)

        if self.battery <= custo_bateria:
            return world.recharger
        

        if self.cargo == 0 and packages:
            alvo, dist = self.dijkstra_multi_target(pos_atual, packages, world)
            dist_ret = self.dijkstra(alvo, world.recharger, world)
            if self.pode_visitar_e_recarregar(dist, dist_ret):
                return alvo
            else:
                return world.recharger

        if self.cargo == len(goals):
            alvo, dist = self.dijkstra_multi_target(pos_atual, goals, world)
            dist_ret = self.dijkstra(alvo, world.recharger, world)
            if self.pode_visitar_e_recarregar(dist, dist_ret):
                return alvo
            else:
                return world.recharger

        if self.cargo > 0 and packages:
            alvo, dist = self.dijkstra_multi_target(pos_atual, pgs, world)
            dist_ret = self.dijkstra(alvo, world.recharger, world)
            if self.pode_visitar_e_recarregar(dist, dist_ret):
                return alvo
            else:
                return world.recharger

        return None

class HybridPlayer(BasePlayer):
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar(self, start, goal, world):
        maze = world.map
        size = world.maze_size
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        close_set = set()
        gscore = {tuple(start): 0}
        fscore = {tuple(start): self.heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[tuple(start)], tuple(start)))
        
        while oheap:
            current = heapq.heappop(oheap)[1]
            if list(current) == goal:
                return gscore[current]  # Retorna custo total

            close_set.add(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                tentative_g = gscore[current] + 1
                if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                    if maze[neighbor[1]][neighbor[0]] == 1:
                        continue
                else:
                    continue
                if neighbor in close_set and tentative_g >= gscore.get(neighbor, 0):
                    continue
                if tentative_g < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                    gscore[neighbor] = tentative_g
                    fscore[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return float('inf')

    def dijkstra_multi_target(self, start, targets, world):
        maze = world.map
        size = world.maze_size
        visited = set()
        dist = {tuple(start): 0}
        heap = [(0, tuple(start))]
        targets_set = set(map(tuple, targets))

        while heap:
            current_dist, current = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)

            if current in targets_set:
                return list(current), current_dist

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                    if maze[neighbor[1]][neighbor[0]] == 0:
                        new_cost = current_dist + 1
                        if new_cost < dist.get(neighbor, float('inf')):
                            dist[neighbor] = new_cost
                            heapq.heappush(heap, (new_cost, neighbor))
        return None, float('inf')

    def escolher_alvo(self, world):
        pos_atual = self.position
        packages = world.packages
        goals = world.goals
        pgs = packages + goals

        custo_retorno = self.astar(pos_atual, world.recharger, world)
        if self.battery <= custo_retorno:
            return world.recharger
        

        if self.cargo == 0 and packages:
            alvo, dist = self.dijkstra_multi_target(pos_atual, packages, world)
            dist_ret = self.astar(alvo, world.recharger, world)
            if self.pode_visitar_e_recarregar(dist, dist_ret):
                return alvo
            else:
                return world.recharger

        if self.cargo == len(goals):
            alvo, dist = self.dijkstra_multi_target(pos_atual, goals, world)
            dist_ret = self.astar(alvo, world.recharger, world)
            if self.pode_visitar_e_recarregar(dist, dist_ret):
                return alvo
            else:
                return world.recharger

        if self.cargo > 0 and packages:
            alvo, dist = self.dijkstra_multi_target(pos_atual, pgs, world)
            dist_ret = self.astar(alvo, world.recharger, world)
            if self.pode_visitar_e_recarregar(dist, dist_ret):
                return alvo
            else:
                return world.recharger

        return None

class EnchacedPlayer(BasePlayer):

    def escolher_alvo(self, world):
        sx, sy = self.position
        dist_ret = abs(world.recharger[0] - sx) + abs(world.recharger[1] - sy)
        # Se não estiver carregando pacote e houver pacotes disponíveis:
        if self.cargo == 0 and world.packages:
            best = None
            best_dist = float('inf')
            for pkg in world.packages:
                d = abs(pkg[0] - sx) + abs(pkg[1] - sy)
                if d < best_dist:
                    best_dist = d
                    best = pkg
            return best if self.pode_visitar_e_recarregar(best_dist, dist_ret) else world.recharger
        else:
            # Se estiver carregando ou não houver mais pacotes, vai para a meta de entrega (se existir)
            if world.goals:
                best = None
                best_dist = float('inf')
                for goal in world.goals:
                    d = abs(goal[0] - sx) + abs(goal[1] - sy)
                    if d < best_dist:
                        best_dist = d
                        best = goal
                return best if self.pode_visitar_e_recarregar(best_dist, dist_ret) else world.recharger
            else:
                return None
# 


class DefaultPlayer(BasePlayer):
    """
    Implementação padrão do jogador.
    Se não estiver carregando pacotes (cargo == 0), escolhe o pacote mais próximo.
    Caso contrário, escolhe a meta (entrega) mais próxima.
    """
    def escolher_alvo(self, world):
        sx, sy = self.position
        # Se não estiver carregando pacote e houver pacotes disponíveis:
        if self.cargo == 0 and world.packages:
            best = None
            best_dist = float('inf')
            for pkg in world.packages:
                d = abs(pkg[0] - sx) + abs(pkg[1] - sy)
                if d < best_dist:
                    best_dist = d
                    best = pkg
            return best
        else:
            # Se estiver carregando ou não houver mais pacotes, vai para a meta de entrega (se existir)
            if world.goals:
                best = None
                best_dist = float('inf')
                for goal in world.goals:
                    d = abs(goal[0] - sx) + abs(goal[1] - sy)
                    if d < best_dist:
                        best_dist = d
                        best = goal
                return best
            else:
                return None

# ==========================
# CLASSE WORLD (MUNDO)
# ==========================
class World:
    def __init__(self, seed=None, player_class=None): 
        if seed is not None:
            random.seed(seed)
        self.seed = seed
        self.maze_size = 50
        self.width = 600
        self.height = 600
        self.block_size = self.width // self.maze_size
        self.map = [[0 for _ in range(self.maze_size)] for _ in range(self.maze_size)]
        self.generate_obstacles()
        self.walls = [(col, row) for row in range(self.maze_size) for col in range(self.maze_size) if self.map[row][col] == 1]
        self.total_items = 6

        self.packages = []
        while len(self.packages) < self.total_items * 2:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.packages:
                self.packages.append([x, y])

        self.goals = []
        while len(self.goals) < self.total_items:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.goals and [x, y] not in self.packages:
                self.goals.append([x, y])

        # Usa player_class passado na construção
        self.player = self.generate_player(player_class) 
        self.recharger = self.generate_recharger()

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Delivery Bot")

        self.package_image = pygame.image.load("images/cargo.png")
        self.package_image = pygame.transform.scale(self.package_image, (self.block_size, self.block_size))

        self.goal_image = pygame.image.load("images/operator.png")
        self.goal_image = pygame.transform.scale(self.goal_image, (self.block_size, self.block_size))

        self.recharger_image = pygame.image.load("images/charging-station.png")
        self.recharger_image = pygame.transform.scale(self.recharger_image, (self.block_size, self.block_size))

        self.wall_color = (100, 100, 100)
        self.ground_color = (255, 255, 255)
        self.player_color = (0, 255, 0)
        self.path_color = (200, 200, 0)

    def generate_obstacles(self):
        """
        Gera obstáculos com sensação de linha de montagem:
         - Cria vários segmentos horizontais curtos com lacunas.
         - Cria vários segmentos verticais curtos com lacunas.
         - Cria um obstáculo em bloco grande (4x4 ou 6x6) simulando uma estrutura de suporte.
        """
        # Barragens horizontais curtas:
        for _ in range(7):
            row = random.randint(5, self.maze_size - 6)
            start = random.randint(0, self.maze_size - 10)
            length = random.randint(5, 10)
            for col in range(start, start + length):
                if random.random() < 0.7:
                    self.map[row][col] = 1

        # Barragens verticais curtas:
        for _ in range(7):
            col = random.randint(5, self.maze_size - 6)
            start = random.randint(0, self.maze_size - 10)
            length = random.randint(5, 10)
            for row in range(start, start + length):
                if random.random() < 0.7:
                    self.map[row][col] = 1

        # Obstáculo em bloco grande: bloco de tamanho 4x4 ou 6x6.
        block_size = random.choice([4, 6])
        max_row = self.maze_size - block_size
        max_col = self.maze_size - block_size
        top_row = random.randint(0, max_row)
        top_col = random.randint(0, max_col)
        for r in range(top_row, top_row + block_size):
            for c in range(top_col, top_col + block_size):
                self.map[r][c] = 1

    def generate_player(self, player_class):  
        while True:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.packages and [x, y] not in self.goals:
                return player_class([x, y])  

    def generate_recharger(self):
        # Coloca o recharger próximo ao centro
        center = self.maze_size // 2
        while True:
            x = random.randint(center - 1, center + 1)
            y = random.randint(center - 1, center + 1)
            if self.map[y][x] == 0 and [x, y] not in self.packages and [x, y] not in self.goals and [x, y] != self.player.position:
                return [x, y]

    def can_move_to(self, pos):
        x, y = pos
        if 0 <= x < self.maze_size and 0 <= y < self.maze_size:
            return self.map[y][x] == 0
        return False

    def draw_world(self, path=None):
        self.screen.fill(self.ground_color)
        # Desenha os obstáculos (paredes)
        for (x, y) in self.walls:
            rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
            pygame.draw.rect(self.screen, self.wall_color, rect)
        # Desenha os locais de coleta (pacotes) utilizando a imagem
        for pkg in self.packages:
            x, y = pkg
            self.screen.blit(self.package_image, (x * self.block_size, y * self.block_size))
        # Desenha os locais de entrega (metas) utilizando a imagem
        for goal in self.goals:
            x, y = goal
            self.screen.blit(self.goal_image, (x * self.block_size, y * self.block_size))
        # Desenha o recharger utilizando a imagem
        if self.recharger:
            x, y = self.recharger
            self.screen.blit(self.recharger_image, (x * self.block_size, y * self.block_size))
        # Desenha o caminho, se fornecido
        if path:
            for pos in path:
                x, y = pos
                rect = pygame.Rect(x * self.block_size + self.block_size // 4,
                                   y * self.block_size + self.block_size // 4,
                                   self.block_size // 2, self.block_size // 2)
                pygame.draw.rect(self.screen, self.path_color, rect)
        # Desenha o jogador (retângulo colorido)
        x, y = self.player.position
        rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
        pygame.draw.rect(self.screen, self.player_color, rect)
        pygame.display.flip()

# ==========================
# CLASSE MAZE: Lógica do jogo e planejamento de caminhos (A*)
# ==========================
class Maze:
    def __init__(self, seed=None, player_class=DefaultPlayer): 
        self.world = World(seed, player_class)  
        self.running = True
        self.score = 0
        self.steps = 0
        self.delay = 10
        self.path = []
        self.num_deliveries = 0

    def heuristic(self, a, b):
        # Distância de Manhattan
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar(self, start, goal):
        maze = self.world.map
        size = self.world.maze_size
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        close_set = set()
        came_from = {}
        gscore = {tuple(start): 0}
        fscore = {tuple(start): self.heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[tuple(start)], tuple(start)))
        while oheap:
            current = heapq.heappop(oheap)[1]
            if list(current) == goal:
                data = []
                while current in came_from:
                    data.append(list(current))
                    current = came_from[current]
                data.reverse()
                return data
            close_set.add(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                tentative_g = gscore[current] + 1
                if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                    if maze[neighbor[1]][neighbor[0]] == 1:
                        continue
                else:
                    continue
                if neighbor in close_set and tentative_g >= gscore.get(neighbor, 0):
                    continue
                if tentative_g < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g
                    fscore[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return []

    def salvar_resultado(self, player_name, seed):
        resultado = {
            "player": player_name,
            "seed": seed,
            "score": self.score,
            "steps": self.steps,
            "deliveries": self.num_deliveries,
            "battery": self.world.player.battery,
            "execution_time": self.tempo_execucao  
        }

        arquivo = "resultados.csv"
        escrever_cabecalho = not os.path.exists(arquivo)
        with open(arquivo, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=resultado.keys())
            if escrever_cabecalho:
                writer.writeheader()
            writer.writerow(resultado)

    def game_loop(self):
        # O jogo termina quando o número de entregas realizadas é igual ao total de itens.
        start_time = time.time() 
        while self.running:
            if self.num_deliveries >= self.world.total_items:
                self.running = False
                break

            # Utiliza a estratégia do jogador para escolher o alvo
            target = self.world.player.escolher_alvo(self.world)
            if target is None:
                self.running = False
                break

            self.path = self.astar(self.world.player.position, target)
            if not self.path:
                print("Nenhum caminho encontrado para o alvo", target)
                self.running = False
                break

            # Segue o caminho calculado
            for pos in self.path:
                self.world.player.position = pos
                self.steps += 1
                # Consumo da bateria: -1 por movimento se bateria >= 0, caso contrário -5
                self.world.player.battery -= 1
                if self.world.player.battery >= 0:
                    self.score -= 1
                else:
                    self.score -= 5
                # Recarrega a bateria se estiver no recharger
                if self.world.recharger and pos == self.world.recharger:
                    self.world.player.battery = 70
                    print("Bateria recarregada!")
                self.world.draw_world(self.path)
                pygame.time.wait(self.delay)

            # Ao chegar ao alvo, processa a coleta ou entrega:
            if self.world.player.position == target:
                # Se for local de coleta, pega o pacote.
                if target in self.world.packages:
                    self.world.player.cargo += 1
                    self.world.packages.remove(target)
                    print("Pacote coletado em", target, "Cargo agora:", self.world.player.cargo)
                # Se for local de entrega e o jogador tiver pelo menos um pacote, entrega.
                elif target in self.world.goals and self.world.player.cargo > 0:
                    self.world.player.cargo -= 1
                    self.num_deliveries += 1
                    self.world.goals.remove(target)
                    self.score += 85
                    print("Pacote entregue em", target, "Cargo agora:", self.world.player.cargo)
            print(f"Passos: {self.steps}, Pontuação: {self.score}, Cargo: {self.world.player.cargo}, Bateria: {self.world.player.battery}, Entregas: {self.num_deliveries}")

        print("Fim de jogo!")
        print("Pontuação final:", self.score)
        print("Total de passos:", self.steps)

        end_time = time.time() 
        self.tempo_execucao = round(end_time - start_time, 2) 
        print("Tempo de execução:", self.tempo_execucao, "segundos") 

        self.salvar_resultado(self.world.player.__class__.__name__, self.world.seed)  

        pygame.quit()

# ==========================
# PONTO DE ENTRADA PRINCIPAL
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delivery Bot: Teste múltiplas estratégias")
    parser.add_argument("--seed", type=int, help="Seed para gerar o mundo")
    
    parser.add_argument("--maps", nargs='+', type=int, help="Lista de seeds para múltiplos mapas")

    parser.add_argument("--multi", action="store_true", help="Rodar todos os players com a mesma seed")
    parser.add_argument("--player", type=str, default="DefaultPlayer", choices=[
        "DefaultPlayer", "DijkstraPlayer", "astarPlayer", "HybridPlayer", "HybridClusterPlayer"
    ], help="Escolha do tipo de player")
    args = parser.parse_args()

    players = {
        "DefaultPlayer": DefaultPlayer,
        'EnchacedPlayer' : EnchacedPlayer,
        "DijkstraPlayer": DijkstraPlayer,
        "astarPlayer": astarPlayer,
        "HybridPlayer": HybridPlayer,
        "HybridClusterPlayer": HybridClusterPlayer  
    }

    if args.maps:
        seeds = args.maps  # lista de seeds fornecida

        if args.multi:
            # Roda todos os players para cada mapa (seed)
            for seed in seeds:
                for name, cls in players.items():
                    print(f"\n======================")
                    print(f"  Rodando {name} na seed {seed}")
                    print(f"======================")
                    maze = Maze(seed=seed, player_class=cls)
                    maze.game_loop()
        else:
            # Roda somente o player escolhido para cada seed
            player_cls = players.get(args.player, DefaultPlayer)
            for seed in seeds:
                print(f"\n======================")
                print(f"  Rodando {args.player} na seed {seed}")
                print(f"======================")
                maze = Maze(seed=seed, player_class=player_cls)
                maze.game_loop()

    else:
        if args.multi:
            for name, cls in players.items():
                print(f"\n======================")
                print(f"  Rodando com {name}")
                print(f"======================")
                maze = Maze(seed=args.seed or 42, player_class=cls)
                maze.game_loop()
        else:
            player_cls = players.get(args.player, DefaultPlayer)
            maze = Maze(seed=args.seed or 42, player_class=player_cls)
            maze.game_loop()


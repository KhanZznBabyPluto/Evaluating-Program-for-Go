#!/usr/bin/env python
import time
import pygame
from game.ui import UI
from game.go import Board, opponent_color
from os.path import join
from argparse import ArgumentParser
from agent.basic_agent import RandomAgent, GreedyAgent
from agent.search.search_agent import AlphaBetaAgent, ExpectimaxAgent
from agent.rl.rl_env import RlEnv
from agent.rl.rl_agent import ApproxQAgent


class Match:
    def __init__(self, agent_black=None, agent_white=None, gui=True, dir_save=None):
        """
        BLACK always has the first move on the center of the board.
        :param agent_black: agent or None(human)
        :param agent_white: agent or None(human)
        :param gui: if show GUI; always true if there are human playing
        :param dir_save: directory to save board image if GUI is shown; no save for None
        """
        self.agent_black = agent_black
        self.agent_white = agent_white

        self.board = Board(next_color='BLACK')

        gui = gui if agent_black and agent_white else True
        self.ui = UI() if gui else None
        self.dir_save = dir_save

        # Metadata
        self.time_elapsed = None

    @property
    def winner(self):
        return self.board.winner

    @property
    def next(self):
        return self.board.next

    @property
    def counter_move(self):
        return self.board.counter_move

    def start(self):
        if self.ui:
            self._start_with_ui()
        else:
            self._start_without_ui()

    def _start_with_ui(self):
        """Start the game with GUI."""
        prev_legal_actions = None
        self.ui.initialize()
        self.time_elapsed = time.time()

        # First move is fixed on the center of board
        first_move = (10, 10)
        self.board.put_stone(first_move, check_legal=False)
        self.ui.draw(first_move, opponent_color(self.board.next))

        # Take turns to play move
        while self.board.winner is None:
            if prev_legal_actions is not None:
                score = self.evaluate_board(prev_legal_actions, 'WHITE')
                print(f'score: {score}')

            if self.board.next == 'BLACK':
                point = self.perform_one_move(self.agent_black)
            else:
                point = self.perform_one_move(self.agent_white)

            # Check if action is legal
            if point not in self.board.legal_actions:
                continue

            # Apply action
            prev_legal_actions = self.board.legal_actions.copy()
            self.board.put_stone(point, check_legal=False)
            # Remove previous legal actions on board
            for action in prev_legal_actions:
                self.ui.remove(action)
            # Draw new point
            self.ui.draw(point, opponent_color(self.board.next))
            # Update new legal actions and any removed groups
            if self.board.winner:
                for group in self.board.removed_groups:
                    for point in group.points:
                        print(f'removed point: {point}')
                        self.ui.remove(point)
                if self.board.end_by_no_legal_actions:
                    print('Game ends early (no legal action is available for %s)' % self.board.next)
            else:
                for action in self.board.legal_actions:
                    self.ui.draw(action, 'BLUE', 8)

        self.time_elapsed = time.time() - self.time_elapsed
        if self.dir_save:
            path_file = join(self.dir_save, 'go_' + str(time.time()) + '.jpg')
            self.ui.save_image(path_file)
            print('Board image saved in file ' + path_file)


    def evaluate_board(self, prev_legal_actions, agent_color):
        """
        Evaluate the current state of the board based on some heuristics.
        :param prev_legal_actions: Previous legal actions before the last move.
        :param agent_color: Color of the agent for which the evaluation is performed.
        :return: A numeric value indicating the evaluation of the board.
        """
        # You can add more heuristics and adjust weights as needed
        score = 0

        # Find the last liberties of black and white groups
        last_liberties = self.find_last_liberties()
        last_liberty_black = last_liberties.get('BLACK')
        last_liberty_white = last_liberties.get('WHITE')

        # Heuristic 1: Encourage capturing opponent stones
        captured_stones = len(prev_legal_actions) - len(self.board.legal_actions)
        score += captured_stones

        # Heuristic 2: Evaluate the efficiency of the last move
        if agent_color == 'BLACK':
            if last_liberty_black:
                score += 1  # Adjust weight as needed
        else:
            if last_liberty_white:
                score += 1  # Adjust weight as needed

        # Heuristic 3: Discourage creating vulnerable groups
        for group in self.board.groups[opponent_color(agent_color)]:
            if len(group.liberties) == 1:
                score -= 1

        # Heuristic 4: Encourage expanding own groups
        for group in self.board.groups[agent_color]:
            score += len(group.liberties)

        return score

    def find_last_liberties(self):
        """
        Find the last liberties of black and white groups.
        :return: A dictionary with 'BLACK' and 'WHITE' keys.
        """
        last_liberties = {'BLACK': set(), 'WHITE': set()}

        for group in self.board.groups['BLACK']:
            # print(f'Black Group : {group}')
            last_liberties['BLACK'].update(group.liberties)

        for group in self.board.groups['WHITE']:
            last_liberties['WHITE'].update(group.liberties)
            # print(f'White Group : {group}')

        return last_liberties


    def _start_without_ui(self):
        """Start the game without GUI. Only possible when no human is playing."""
        # First move is fixed on the center of board
        self.time_elapsed = time.time()
        first_move = (10, 10)
        self.board.put_stone(first_move, check_legal=False)

        # Take turns to play move
        while self.board.winner is None:
            if self.board.next == 'BLACK':
                point = self.perform_one_move(self.agent_black)
            else:
                point = self.perform_one_move(self.agent_white)

            # Apply action
            self.board.put_stone(point, check_legal=False)  # Assuming agent always gives legal actions

        if self.board.end_by_no_legal_actions:
            print('Game ends early (no legal action is available for %s)' % self.board.next)

        self.time_elapsed = time.time() - self.time_elapsed

    def perform_one_move(self, agent):
        if agent:
            return self._move_by_agent(agent)
        else:
            return self._move_by_human()

    def _move_by_agent(self, agent):
        if self.ui:
            pygame.time.wait(100)
            pygame.event.get()
        return agent.get_action(self.board)

    def _move_by_human(self):
        while True:
            pygame.time.wait(100)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and self.ui.outline.collidepoint(event.pos):
                        x = int(round(((event.pos[0] - 5) / 40.0), 0))
                        y = int(round(((event.pos[1] - 5) / 40.0), 0))
                        point = (x, y)
                        stone = self.board.exist_stone(point)
                        if not stone:
                            return point


def get_args():
    parser = ArgumentParser('Mini Go Game')
    parser.add_argument('-b', '--agent_black', default=None,
                        help='possible agents: random; greedy; minimax; expectimax, approx-q; DEFAULT is None (human)')
    parser.add_argument('-w', '--agent_white', default=None,
                        help='possible agents: random; greedy; minimax; expectimax, approx-q; DEFAULT is None (human)')
    parser.add_argument('-d', '--search_depth', type=int, default=1,
                        help='the search depth for searching agents if applicable; DEFAULT is 1')
    parser.add_argument('-g', '--gui', type=bool, default=True,
                        help='if show GUI; always true if human plays; DEFAULT is True')
    parser.add_argument('-s', '--dir_save', default=None,
                        help='if not None, save the image of last board state to this directory; DEFAULT is None')
    return parser.parse_args()


def get_agent(str_agent, color, depth):
    if str_agent is None:
        return None
    str_agent = str_agent.lower()
    if str_agent == 'none':
        return None
    elif str_agent == 'random':
        return RandomAgent(color)
    elif str_agent == 'greedy':
        return GreedyAgent(color)
    elif str_agent == 'minimax':
        return AlphaBetaAgent(color, depth=depth)
    elif str_agent == 'expectimax':
        return ExpectimaxAgent(color, depth=depth)
    elif str_agent == 'approx-q':
        agent = ApproxQAgent(color, RlEnv())
        agent.load('agent/rl/ApproxQAgent.npy')
        return agent
    else:
        raise ValueError('Invalid agent for ' + color)


def main():
    args = get_args()
    depth = args.search_depth
    agent_black = get_agent(args.agent_black, 'BLACK', depth)
    agent_white = get_agent(args.agent_white, 'WHITE', depth)
    gui = args.gui
    dir_save = args.dir_save

    print('Agent for BLACK: ' + (str(agent_black) if agent_black else 'Human'))
    print('Agent for WHITE: ' + (str(agent_white) if agent_white else 'Human'))
    if dir_save:
        print('Directory to save board image: ' + dir_save)

    match = Match(agent_black=agent_black, agent_white=agent_white, gui=gui, dir_save=dir_save)

    print('Match starts!')
    match.start()

    print(match.winner + ' wins!')
    print('Match ends in ' + str(match.time_elapsed) + ' seconds')
    print('Match ends in ' + str(match.counter_move) + ' moves')


if __name__ == '__main__':
    # match = Match()
    # match = Match(agent_black=RandomAgent('BLACK'))
    # match = Match(agent_black=RandomAgent('BLACK'), agent_white=RandomAgent('WHITE'), gui=True)
    # match = Match(agent_black=RandomAgent('BLACK'), agent_white=RandomAgent('WHITE'), gui=False)
    # match.start()
    # print(match.winner + ' wins!')
    # print('Match ends in ' + str(match.time_elapsed) + ' seconds')
    # print('Match ends in ' + str(match.counter_move) + ' moves')
    main()

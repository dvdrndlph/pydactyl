__author__ = 'David Randolph'
# Copyright (c) 2014 David A. Randolph.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

import time

class GraphNode:
    START = 'Start'
    END = 'End'

    def __init__(self, terminal=None):
        if terminal and terminal != GraphNode.START and terminal != GraphNode.END:
            raise Exception('Bad terminal setting for GraphNode.')
        self.terminal = terminal
        self.next_nodes = []
        self.prior_nodes = []

    def is_start(self):
        if self.terminal and self.terminal == GraphNode.START:
            return True
        return False

    def is_end(self):
        if self.terminal and self.terminal == GraphNode.END:
            return True
        return False

    # Should be overridden by derived class
    def can_transition_to(self, next_node):
        if self.is_start() or next_node.is_end():
            return True
        if self.is_end():
            return False

    def connect_from(self, prior_node):
        self.prior_nodes.append(prior_node)

    def disconnect(self, next_node=None):
        if self.is_start():
            # raise Exception('Cannot disconnect: child nodes present.')
            return

        if self.next_nodes:
            self.next_nodes.remove(next_node)

        if not self.next_nodes:
            for prior_node in self.prior_nodes:
                prior_node.disconnect(next_node=self)
            self.prior_nodes = []

    # Override in derived class to enforce restrictions.
    def can_transition_to(self, next_node):
        return True

    def connect_to(self, next_node):
        if self.can_transition_to(next_node):
            if not next_node in self.next_nodes:
                self.next_nodes.append(next_node)
                next_node.connect_from(self)
                return True
            return False
        return False

    def dump(self):
        print str(self)

all: build run

max: maxbuild run

build: chat.cpp
	g++ -std=c++17 chat.cpp -I/usr/include/python3.10 -lpython3.10 -lnlopt  -o path_planner

maxbuild: chat.cpp
	g++ -std=c++17 chat.cpp -I/usr/include/python3.10 -lpython3.10 -lnlopt -O3 -o path_planner

clean: ./path_planner
	rm -f path_planner

run: ./path_planner
	./path_planner
objs = ../neuron.o ../mad/matrix/matrix.o
progs  = create_net_test feedforward_test

CFLAGS = -I../ -I../mad/matrix/
LDLIBS = -lm
CC = gcc

all:	$(progs)

clean:
	rm $(objs)  $(progs)

nums_test: $(objs)
save_test: $(objs)
faces_test: $(objs)
myface_test: $(objs)
create_net_test: $(objs)
feedforward_test: $(objs)

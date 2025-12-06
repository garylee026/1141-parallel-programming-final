CC = cc
CFLAGS = -O2
TARGET = mnist
SRCS = mnist.c cnn.c

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) -lm

clean:
	rm -f $(TARGET)

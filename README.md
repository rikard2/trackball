1. installera opencv3, brew install opencv3 --with-ffmpeg (brew info opencv3 för att lista alla flaggor)
2. kompilera med: g++ $(pkg-config --cflags --libs /usr/local/Cellar/opencv3/3.2.0/lib/pkgconfig/opencv.pc) timetrax.cpp -o timetrax
3. ???
4. ./timetrax
5  profit!

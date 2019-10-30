# istrazivanje_podataka1_seminarski
Seminarski rad u okviru kursa Istrazivanje Podataka 1

U radu je predstavljen osnovni pristup obrade teksta nad [imdb](https://ai.stanford.edu/~amaas/data/sentiment/) skupom podataka.
Jedan od problema(kasnije uocen) koji nije uzet u obzir je (curenje podataka) data leakage, tj prvo je vrsena transformacija teksta u tf-idf reprezentaciju
pa zatim podela na trening i test skup. Ovim su informacije iz test skupa bile poznate u fazi treniranja modela.

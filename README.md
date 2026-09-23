# READ ME STAGE INRIA

## Stage M1 — Simulation de systèmes Hamiltoniens
### Émission cyclotron

3 juin – 17 juillet 2025, Montpellier

Encadré par Dr. Hélène Mathis et Dr. Benoit Jouault, avec l'aide de Frédéric Teppe

## 📁 Contenu du dépôt

* `Rapport_de_stage_M1` : Rapport Latex qui explique la première approche réalisée lors du stage, ainsi que le code final rendu à l'issue du stage.

* `Jouet` / `Bande_parabolique` / `Bande_Fermions_et_Graphene_G0` / `Bande_graphene_Sergei(lineaire)` : Ces notebooks représentent chacun un système Hamiltonien différent à étudier. Le premier étant le cas "Jouet", qui est le problème le plus trivial et permet de comprendre les enjeux du stage.

* `streaming_example` : Code qui permet de simuler la trajectoire des particules selon les bandes choisies.
* `Explicit symplectic approximation...` : PDF  qui permet de comprendre la méthode d'intégration utilisée dans le code précédent.

Les librairies classiques sont utilisées, à part PyHamSys qui devra sûrement être installée et qui regroupe des méthodes liées à la résolution de systèmes Hamiltoniens.

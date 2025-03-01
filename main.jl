using ProgressMeter
using Logging
using Dates
include("modules/kmeans")  # Inclusion du fichier K-means
include("modules/gmm")  # Inclusion du fichier GMM
include("modules/comparateur_gmm")  # Inclusion du comparateur GMM

# 📁 Créer des dossiers de sortie si inexistants
mkpath("output/logs")
mkpath("output/results/kmeans")
mkpath("output/results/gmm")

# 📝 Initialiser le journal de logs
logfile = open("output/logs/log_$(Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")).txt", "w")
global_logger(ConsoleLogger(logfile))

# 📌 Liste des étapes du pipeline
tasks = [
    ("Exécution du clustering K-means", () -> run_kmeans("data/data_pfe_n1000_L10.csv", 3)),
    ("Détermination du nombre optimal de clusters pour GMM", () -> run_comparateur_gmm("data/data_pfe_n1000_L10.csv", 1, 10)),
    ("Exécution du clustering GMM", () -> begin
        optimal_k = run_comparateur_gmm("data/data_pfe_n1000_L10.csv", 1, 10)
        run_gmm("data/data_pfe_n1000_L10.csv", optimal_k)
    end)
]

# 🚀 Exécution avec barre de progression
@showprogress "Exécution du pipeline..." for (desc, func) in tasks
    try
        println("\n🔄 $desc...")
        func()  # Appeler la fonction correspondante
        println("✅ Terminé : $desc")
    catch e
        @error "❌ Erreur lors de : $desc" exception=(e, catch_backtrace())
    end
end

println("\n🎉 Exécution terminée ! Résultats disponibles dans /output/results")
close(logfile)

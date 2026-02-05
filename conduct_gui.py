import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.colors as mcolors

# --- CONFIGURATION ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class MoteurPhysique:
    """
    Moteur conforme à l'équation (4) du TP :
    d²T/dx² - m² * (T - T_inf) = 0
    avec m² = (4 * h) / (lambda * D)
    """
    
    @staticmethod
    def interpoler_h(x, L, hA, hM, hB):
        # Interpolation parabolique pour le mode variable (Section 3.3)
        xp = [0, L/2, L]
        fp = [hA, hM, hB]
        coeffs = np.polyfit(xp, fp, 2)
        h_vals = np.polyval(coeffs, x)
        return np.maximum(h_vals, 0)

    @staticmethod
    def resoudre(L, D, Lambda, nodes, Ta, Tb, T_inf, h_params, mode="constant"):
        N = nodes + 2
        x = np.linspace(0, L, N)
        dx = x[1] - x[0]
        
        # 1. Définition de h(x)
        if mode == "constant":
            h_val = h_params
            h_x = np.full_like(x, h_val)
        else:
            hA, hM, hB = h_params
            h_x = MoteurPhysique.interpoler_h(x, L, hA, hM, hB)

        # 2. Calcul du coefficient alpha (m² * dx²)
        # Equation: T(i-1) - (2 + alpha)T(i) + T(i+1) = - alpha * T_inf
        # alpha = (4 * h * dx^2) / (lambda * D)
        
        # Sécurité pour éviter division par zéro
        if D <= 0 or Lambda <= 0:
            return None, None
            
        alpha = (4 * h_x * (dx**2)) / (Lambda * D)
        
        # 3. Système Matriciel (TDMA)
        main_diag = -1 * (2 + alpha[1:-1])
        off_diag = 1 * np.ones(nodes - 1)
        
        A = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
        
        B = -1 * alpha[1:-1] * T_inf
        
        # Conditions aux limites
        B[0] -= Ta
        B[-1] -= Tb
        
        try:
            T_internal = np.linalg.solve(A, B)
            T_final = np.concatenate(([Ta], T_internal, [Tb]))
            return x, T_final
        except np.linalg.LinAlgError:
            return None, None

class AppConductTP(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("TP1 : Transfert Thermique (Simulateur Complet)")
        self.geometry("1300x850")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Stockage des courbes pour comparaison
        # Liste de dictionnaires : {'x': x, 'y': y, 'label': label}
        self.courbes_stockees = [] 

        self.setup_sidebar()
        self.setup_main_area()

    def setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        # Titre
        ctk.CTkLabel(self.sidebar, text="DONNÉES DU TP", font=("Arial", 20, "bold"), text_color="#3B8ED0").pack(pady=(20, 10))

        # --- INPUTS (Ordre du TP) ---
        self.inputs = {}
        
        # Fonction utilitaire pour créer les inputs
        def add_input(label_text, default_val, tooltip=""):
            frm = ctk.CTkFrame(self.sidebar, fg_color="transparent")
            frm.pack(fill="x", padx=15, pady=2)
            lbl = ctk.CTkLabel(frm, text=label_text, anchor="w", font=("Arial", 12))
            lbl.pack(fill="x")
            entry = ctk.CTkEntry(frm, height=28)
            entry.pack(fill="x")
            entry.insert(0, default_val)
            self.inputs[label_text] = entry

        # 1. Géométrie & Matériau
        add_input("Diamètre de la barre (m)", "0.01")
        add_input("Conductivité λ (Cuivre) W/m.K", "385.0") # Ajouté pour la physique exacte
        add_input("Nombre de nœuds internes", "19")
        
        ctk.CTkFrame(self.sidebar, height=2, fg_color="gray40").pack(fill="x", padx=20, pady=10)

        # 2. Températures
        add_input("Température Extrémité A (°C)", "25.0")
        add_input("Température Extrémité B (°C)", "1000.0")
        add_input("Température Ambiante (°C)", "25.0")

        ctk.CTkFrame(self.sidebar, height=2, fg_color="gray40").pack(fill="x", padx=20, pady=10)

        # 3. Convection (h) avec Onglets
        ctk.CTkLabel(self.sidebar, text="COEFFICIENT h", font=("Arial", 14, "bold")).pack(pady=5)
        
        self.tab_h = ctk.CTkTabview(self.sidebar, height=140)
        self.tab_h.pack(fill="x", padx=15)
        
        # Onglet h Constant
        self.tab_h.add("Constant (h)")
        self.input_h_cst = ctk.CTkEntry(self.tab_h.tab("Constant (h)"), placeholder_text="Valeur de h")
        self.input_h_cst.pack(pady=20)
        self.input_h_cst.insert(0, "5.0")
        
        # Onglet h Variable (A, M, B)
        self.tab_h.add("Variable (A-M-B)")
        frm_var = ctk.CTkFrame(self.tab_h.tab("Variable (A-M-B)"), fg_color="transparent")
        frm_var.pack(pady=5)
        
        self.hA = ctk.CTkEntry(frm_var, width=50); self.hA.grid(row=0, column=0, padx=2); self.hA.insert(0, "3")
        self.hM = ctk.CTkEntry(frm_var, width=50); self.hM.grid(row=0, column=1, padx=2); self.hM.insert(0, "9")
        self.hB = ctk.CTkEntry(frm_var, width=50); self.hB.grid(row=0, column=2, padx=2); self.hB.insert(0, "3")
        ctk.CTkLabel(frm_var, text="h(A)      h(M)      h(B)", font=("Arial", 10)).grid(row=1, column=0, columnspan=3)

        # --- ACTION BUTTONS ---
        self.btn_run = ctk.CTkButton(self.sidebar, text="▶ LANCER SIMULATION", font=("Arial", 13, "bold"), 
                                     fg_color="#1F6AA5", hover_color="#144870", height=40, command=self.calculer)
        self.btn_run.pack(padx=15, pady=(20, 10), fill="x")

        # Bouton Reset & Export cote à cote
        frm_btn = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        frm_btn.pack(padx=15, pady=5, fill="x")
        
        self.btn_reset = ctk.CTkButton(frm_btn, text="🗑️ Effacer Tout", width=100, fg_color="#C0392B", hover_color="#922B21", command=self.reset_graph)
        self.btn_reset.pack(side="left", padx=(0, 5), expand=True, fill="x")
        
        self.btn_save = ctk.CTkButton(frm_btn, text="💾 Excel", width=100, fg_color="#27AE60", hover_color="#1E8449", command=self.exporter)
        self.btn_save.pack(side="right", padx=(5, 0), expand=True, fill="x")

    def setup_main_area(self):
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Matplotlib Figure
        self.fig = plt.figure(figsize=(8, 8), dpi=100)
        self.fig.patch.set_facecolor('#242424')
        
        gs = self.fig.add_gridspec(2, 1, height_ratios=[4, 1])
        self.ax = self.fig.add_subplot(gs[0])
        self.ax_bar = self.fig.add_subplot(gs[1])
        
        self.configurer_axes()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.main_frame)
        self.toolbar.update()

    def configurer_axes(self):
        # Reset Axe Courbe
        self.ax.clear()
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.set_xlabel("Position x (m)", color='white')
        self.ax.set_ylabel("Température T (°C)", color='white')
        self.ax.set_title("Profil de Température", color='white', fontsize=14)
        
        # Reset Axe Barre
        self.ax_bar.clear()
        self.ax_bar.set_facecolor('#242424')
        self.ax_bar.axis('off')

    def get_val(self, key):
        return float(self.inputs[key].get())

    def calculer(self):
        try:
            # Récupération Inputs
            D = self.get_val("Diamètre de la barre (m)")
            Lambda = self.get_val("Conductivité λ (Cuivre) W/m.K")
            nodes = int(self.inputs["Nombre de nœuds internes"].get())
            Ta = self.get_val("Température Extrémité A (°C)")
            Tb = self.get_val("Température Extrémité B (°C)")
            T_inf = self.get_val("Température Ambiante (°C)")
            L = 1.0 # Fixe selon le TP ou ajoutable en input
            
            mode = self.tab_h.get()
            
            if mode == "Constant (h)":
                h_val = float(self.input_h_cst.get())
                x, T = MoteurPhysique.resoudre(L, D, Lambda, nodes, Ta, Tb, T_inf, h_val, "constant")
                label = f"h={h_val}"
            else:
                hA = float(self.hA.get())
                hM = float(self.hM.get())
                hB = float(self.hB.get())
                x, T = MoteurPhysique.resoudre(L, D, Lambda, nodes, Ta, Tb, T_inf, (hA, hM, hB), "variable")
                label = f"h_var({hA}-{hM}-{hB})"
            
            if x is None:
                raise ValueError("Erreur mathématique (Diamètre ou Lambda nul ?)")

            # Ajout à la liste pour comparaison
            self.courbes_stockees.append({'x': x, 'y': T, 'label': label})
            
            # Mise à jour graphique
            self.update_graph(x, T, Ta, Tb)
            
        except ValueError as e:
            messagebox.showerror("Erreur", "Vérifiez vos données numériques.\n" + str(e))

    def update_graph(self, last_x, last_T, Ta, Tb):
        # 1. Redessiner TOUTES les courbes stockées (Superposition)
        self.ax.clear()
        self.configurer_axes() # Remet le style
        
        # Palette de couleurs automatique
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(self.courbes_stockees))))
        
        for i, data in enumerate(self.courbes_stockees):
            self.ax.plot(data['x'], data['y'], marker='o', markersize=3, 
                         linewidth=2, label=data['label'], color=colors[i])
        
        self.ax.legend(facecolor='#333', labelcolor='white')
        
        # 2. Dessiner la barre (Heatmap) pour la DERNIÈRE simulation seulement
        self.ax_bar.clear()
        self.ax_bar.axis('off')
        
        # Création Heatmap
        im = self.ax_bar.imshow(last_T.reshape(1, -1), aspect='auto', cmap='inferno', 
                                extent=[0, 1, 0, 1])
        
        # Annotations
        self.ax_bar.text(0, 0.5, f"{Ta}°C", color='white', fontweight='bold', ha='left', va='center')
        self.ax_bar.text(1, 0.5, f"{Tb}°C", color='white', fontweight='bold', ha='right', va='center')
        self.ax_bar.set_title("Visualisation Thermique (Dernier calcul)", color='white', fontsize=10)
        
        self.fig.tight_layout()
        self.canvas.draw()

    def reset_graph(self):
        self.courbes_stockees = []
        self.configurer_axes()
        self.ax_bar.clear()
        self.ax_bar.axis('off')
        self.canvas.draw()

    def exporter(self):
        if not self.courbes_stockees:
            messagebox.showwarning("Vide", "Aucune donnée à sauvegarder.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Enregistrer sous...",
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")]
        )
        
        if not filepath: return

        # Création DataFrame
        # On suppose que x est le même pour tous (sauf si on change nodes/L entre temps)
        # Pour être sûr, on exporte la dernière courbe calculée ou un tableau combiné
        
        # Export combiné intelligent
        data = {}
        # On prend le x de la première courbe comme référence (si nodes change, ça peut décaler, 
        # mais on simplifie ici pour l'usage TP où nodes reste souvent 19)
        ref_x = self.courbes_stockees[0]['x']
        data["Position (m)"] = ref_x
        
        for i, courbe in enumerate(self.courbes_stockees):
            col_name = f"T(°C) [{courbe['label']}]"
            # Si la taille diffère (changement de noeuds), on pad avec NaN
            if len(courbe['y']) != len(ref_x):
                messagebox.showwarning("Attention", "Vous avez varié le nombre de nœuds. L'export peut être décalé.")
                # Fallback simple : on exporte juste la dernière
                df = pd.DataFrame({"Position": courbe['x'], "Temperature": courbe['y']})
            else:
                data[col_name] = courbe['y']
        
        df = pd.DataFrame(data)

        try:
            if filepath.endswith(".csv"):
                df.to_csv(filepath, index=False)
            else:
                # Vérification openpyxl
                try:
                    import openpyxl
                    df.to_excel(filepath, index=False)
                except ImportError:
                    messagebox.showerror("Erreur Module", "Module 'openpyxl' manquant.\nSauvegarde forcée en CSV.")
                    df.to_csv(filepath.replace(".xlsx", ".csv"), index=False)
            
            messagebox.showinfo("Succès", "Fichier sauvegardé !")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

if __name__ == "__main__":
    app = AppConductTP()
    app.mainloop()
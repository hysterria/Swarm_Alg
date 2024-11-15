import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Целевая функция
def objective_function(x, y):
    return (x - 2) ** 4 + (x - 2 * y) ** 2


# Класс Частицы
class Particle:
    def __init__(self, bounds):
        self.position = np.array([np.random.uniform(*b) for b in bounds])
        self.velocity = np.zeros(len(bounds))
        self.best_position = self.position.copy()
        self.best_value = objective_function(*self.position)

    def update_velocity(self, global_best, inertia, cognitive, social, use_inertia):
        r1, r2 = np.random.rand(), np.random.rand()
        cognitive_component = cognitive * r1 * (self.best_position - self.position)
        social_component = social * r2 * (global_best - self.position)

        # Учитываем инерцию веса, если включен соответствующий режим
        if not use_inertia:
            self.velocity = inertia * self.velocity + cognitive_component + social_component
        else:
            self.velocity = cognitive_component + social_component

    def update_position(self, bounds):
        self.position += self.velocity
        for i, bound in enumerate(bounds):
            if self.position[i] < bound[0]:
                self.position[i] = bound[0]
            elif self.position[i] > bound[1]:
                self.position[i] = bound[1]
        value = objective_function(*self.position)
        if value < self.best_value:
            self.best_value = value
            self.best_position = self.position.copy()


# Класс PSO
class PSO:
    def __init__(self, num_particles, bounds, inertia, cognitive, social, use_inertia):
        self.particles = [Particle(bounds) for _ in range(num_particles)]
        self.global_best_position = min(self.particles, key=lambda p: p.best_value).best_position
        self.global_best_value = objective_function(*self.global_best_position)
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.bounds = bounds
        self.use_inertia = use_inertia

    # Измененный метод optimize в классе PSO
    def optimize(self, iterations, app):
        for _ in range(iterations):
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.inertia, self.cognitive, self.social,
                                         self.use_inertia)
                particle.update_position(self.bounds)
                if particle.best_value < self.global_best_value:
                    self.global_best_value = particle.best_value
                    self.global_best_position = particle.best_position
            # Обновление графика после каждого шага
            app.plot_particles()
            app.root.update_idletasks()  # Перерисовка интерфейса


# Интерфейс
class PSOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Роевой интеллект")

        # Параметры
        params_frame = tk.LabelFrame(root, text="Параметры")
        params_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

        tk.Label(params_frame, text="Функция:").grid(row=0, column=0, sticky="w")
        self.function_var = tk.StringVar(value="(x-2)^4 + (x-2y)^2")
        tk.Entry(params_frame, textvariable=self.function_var, state='readonly').grid(row=0, column=1, sticky="ew")

        tk.Label(params_frame, text="Коэфф. текущей скорости:").grid(row=1, column=0, sticky="w")
        self.inertia_entry = tk.Entry(params_frame)
        self.inertia_entry.grid(row=1, column=1)
        self.inertia_entry.insert(0, "0.5")

        tk.Label(params_frame, text="Коэфф. собственного лучшего значения:").grid(row=2, column=0, sticky="w")
        self.cognitive_entry = tk.Entry(params_frame)
        self.cognitive_entry.grid(row=2, column=1)
        self.cognitive_entry.insert(0, "1.5")

        tk.Label(params_frame, text="Коэфф. глобального лучшего значения:").grid(row=3, column=0, sticky="w")
        self.social_entry = tk.Entry(params_frame)
        self.social_entry.grid(row=3, column=1)
        self.social_entry.insert(0, "1.5")

        tk.Label(params_frame, text="Количество частиц:").grid(row=4, column=0, sticky="w")
        self.num_particles_entry = tk.Entry(params_frame)
        self.num_particles_entry.grid(row=4, column=1)
        self.num_particles_entry.insert(0, "30")

        # Добавляем кнопку для включения/выключения инерции веса
        self.use_inertia_var = tk.BooleanVar(value=True)
        self.use_inertia_check = tk.Checkbutton(params_frame, text="Использовать инерцию веса",
                                                variable=self.use_inertia_var)
        self.use_inertia_check.grid(row=5, column=0, columnspan=2, sticky="w")

        # Управление
        control_frame = tk.LabelFrame(root, text="Управление")
        control_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")

        tk.Button(control_frame, text="Создать частицы", command=self.create_particles).grid(row=0, column=0,
                                                                                             columnspan=2, pady=5)

        tk.Label(control_frame, text="Количество итераций:").grid(row=1, column=0)
        self.iterations_entry = tk.Entry(control_frame)
        self.iterations_entry.grid(row=1, column=1)
        self.iterations_entry.insert(0, "100")

        tk.Button(control_frame, text="Рассчитать", command=self.run_optimization).grid(row=2, column=0, columnspan=2,
                                                                                        pady=5)

        # Результаты
        results_frame = tk.LabelFrame(root, text="Результаты")
        results_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        self.result_text = tk.Text(results_frame, width=40, height=5)
        self.result_text.grid(row=0, column=0, padx=5, pady=5)

        # График
        self.figure, self.ax = plt.subplots()
        self.ax.set_title("Решения")
        self.ax.grid(True)

        plot_frame = tk.LabelFrame(root, text="График")
        plot_frame.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_particles(self):
        self.num_particles = int(self.num_particles_entry.get())
        self.bounds = [(-500, 500), (-500, 500)]
        self.inertia = float(self.inertia_entry.get())
        self.cognitive = float(self.cognitive_entry.get())
        self.social = float(self.social_entry.get())

        # Создаем PSO с текущими параметрами
        self.pso = PSO(
            num_particles=self.num_particles,
            bounds=self.bounds,
            inertia=self.inertia,
            cognitive=self.cognitive,
            social=self.social,
            use_inertia=self.use_inertia_var.get()
        )

        # Обновляем текст в выводе
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Частицы созданы.\n")
        self.plot_particles()  # Первое отображение частиц на графикеастицы созданы.\n")
        self.plot_particles()

    def run_optimization(self):

        # Переключение режима инерции
        use_inertia = self.use_inertia_var.get()

        # Создаем новый объект PSO с текущими параметрами и режимом инерции
        self.pso = PSO(
            num_particles=int(self.num_particles_entry.get()),
            bounds=[(-500, 500), (-500, 500)],
            inertia=float(self.inertia_entry.get()),
            cognitive=float(self.cognitive_entry.get()),
            social=float(self.social_entry.get()),
            use_inertia=use_inertia
        )

        # Запуск оптимизации
        iterations = int(self.iterations_entry.get())
        self.pso.optimize(iterations, self)

        # Вывод результата
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Лучшее решение:\nX[0] = {self.pso.global_best_position[0]:.4f}\n"
                                        f"X[1] = {self.pso.global_best_position[1]:.4f}\n"
                                        f"Значение функции: {self.pso.global_best_value:.4f}\n")
        self.plot_particles()

    def plot_particles(self):
        self.ax.clear()
        self.ax.set_title("Решения")
        self.ax.set_xlim(-500, 500)
        self.ax.set_ylim(-500, 500)

        positions = np.array([p.position for p in self.pso.particles])
        self.ax.scatter(positions[:, 0], positions[:, 1], s=1)
        self.canvas.draw()


# Запуск интерфейса
root = tk.Tk()
app = PSOApp(root)
root.mainloop()





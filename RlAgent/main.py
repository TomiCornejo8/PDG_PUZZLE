import gymEnv as env


mainenv=env.DungeonEnv()
model =env.createModel(mainenv)

# Entrenamiento del modelo
model.learn(total_timesteps=500000)


model.save("ppo_dungeon")

# Cargar y evaluar el modelo
model = env.getModel(mainenv)

obs, _ = mainenv.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, truncated, info = mainenv.step(action)
    mainenv.render()
    if done:
        obs, _ = mainenv.reset()


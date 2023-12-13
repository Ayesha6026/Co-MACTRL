import numpy as np
from typing import Tuple
from pprint import pprint
import matplotlib.pyplot as plt

class InfluenceMap():
    """ An influence map is an abstraction of a global state.

        Attributes:
            x_influenceMap  (int):   The X dimension of the influence map.
            y_influenceMap  (int):   The Y dimension of the influence map.
            x_positionScale (float): The position scale of X axis.
            y_positionScale (float): The position scale of Y axis.
            env             (obj):   The environment used to create the influence Map.
            env_state_function (fn): The function defined in the environment that returns the global state.
            display_image   (bool):  Determines if the influence map should be displayed as an image.
            save_image      (bool):  Indicates if an image of the influence map should be saved.
            unitAttackRanges(dict):  A dict containing various units attack ranges. Key is unit id, value is range.
            unitNames       (dict):  Key is unit_id, value is unit_name.
            allyFeatureCount(int):   Represents how many features each ally unit has within the global state.
            enemyFeatureCount(int):  Represents how many features each enemy unit has within the global state.
            influenceMap    (ndarray): The actual influence map associated with this class.
            allyFeatureCount(int):    

    """
    def __init__(self, X = 64, Y = 64, normalize=True, display_image=False, save_image=False):
        self.x_influenceMap = X
        self.y_influenceMap = Y

        # All of these are set during the setup_env_and_influence_map() call.
        self.x_positionScale = 0
        self.y_positionScale = 0
        self.env = None
        self.env_state_function = None
        self.display_image = display_image
        self.save_image = save_image
        self.unitAttackRanges = {}
        self.unitNames = {}
        self.allyFeatureCount = 0
        self.enemyFeatureCount = 0
        self.normalize = normalize
        self.sight_range = 18
        self.booleanInfluenceMap = False

        self.influenceMap = np.zeros(
            (self.x_influenceMap, self.y_influenceMap))

        if self.display_image or self.save_image:
            self.plt = plt
            self.plt.figure(figsize=(10, 10))
            self.influence_map_image = self.plt.imshow(self.influenceMap.T * -1.0, cmap="bwr", interpolation="nearest", origin='lower')
            self.plt.clim(-3, 3)
            self.plt.axis("off")
            self.plt.tight_layout()
        
        if self.display_image:
            self.plt.show(block=False)
            self.plt.pause(0.001)

    def get_combined_local_obs_and_influence_map(self, local_obs,agent_id):
        """ Returns an array conatining a concatenated structure of [local_obs, influence_map]
            for a single agent.

            Args:
                local_obs (ndarray): An array containing the local observations of a single agent.

            Returns:
                An ndarray of concatenated local_obs and influence map.
        """
        if self.normalize:
            self.influenceMap = np.where(self.influenceMap, (2.*(self.influenceMap - np.min(self.influenceMap))/np.ptp(self.influenceMap)-1), self.influenceMap)

        state = np.concatenate((self.influenceMap.flatten(), self.allies[agent_id]["agentObservedLocalInfluenceMAIM"].flatten()), axis=0)
        # state = np.expand_dims(state, axis=0) # Don't need this for now.

        return state
    
    def get_combined_enemy_obs_and_influence_map(self, local_obs,agent_id):
        """ Returns an array conatining a concatenated structure of [local_obs, influence_map]
            for a single agent.

            Args:
                local_obs (ndarray): An array containing the local observations of a single agent.

            Returns:
                An ndarray of concatenated local_obs and influence map.
        """
        if self.normalize:
            self.influenceMap = np.where(self.influenceMap, (2.*(self.influenceMap - np.min(self.influenceMap))/np.ptp(self.influenceMap)-1), self.influenceMap)

        state = np.concatenate((self.influenceMap.flatten(), self.enemies[agent_id]["agentObservedLocalInfluenceMAIM"].flatten()), axis=0)
        # state = np.expand_dims(state, axis=0) # Don't need this for now.

        return state

    def setup_env_and_influence_map(self, env):
        """ This function sets the environment and some properties of the influence map.
            It only needs to be run once.
            
            Args:
                env (obj): An SC2 env object.
        """
        self.env = env

        self.x_positionScale = self.x_influenceMap / self.env.map_x
        self.y_positionScale = self.y_influenceMap / self.env.map_y

        self.unitAttackRanges = {
            self.env.baneling_id: 1 + 0.25,
            self.env.colossus_id: 1 + 7,
            self.env.hydralisk_id: 1 + 5,
            self.env.marauder_id: 1 + 6,
            self.env.marine_id: 1 + 5,
            self.env.medivac_id: 1 + 4,
            self.env.stalker_id: 1 + 6,
            self.env.zealot_id: 1 + 0.1,
            self.env.zergling_id: 1 + 0.1,
        }

        # Now that we have the allied units, let's add the enemy units
        self.unitAttackRanges[9] = self.unitAttackRanges[self.env.baneling_id]
        self.unitAttackRanges[4] = self.unitAttackRanges[self.env.colossus_id]
        self.unitAttackRanges[51] = self.unitAttackRanges[self.env.marauder_id]
        self.unitAttackRanges[48] = self.unitAttackRanges[self.env.marine_id]
        self.unitAttackRanges[54] = self.unitAttackRanges[self.env.medivac_id]
        self.unitAttackRanges[74] = self.unitAttackRanges[self.env.stalker_id]
        self.unitAttackRanges[73] = self.unitAttackRanges[self.env.zealot_id]
        self.unitAttackRanges[105] = self.unitAttackRanges[self.env.zergling_id]
        # Add the spine crawler in as a potential enemy
        self.unitAttackRanges[98] = 1 + 7

        # Add unit type mapping
        self.standardEnemyUnitTypes = {
            9: self.env.baneling_id,
            4: self.env.colossus_id,
            51: self.env.marauder_id,
            48: self.env.marine_id,
            2005: self.env.stalker_id,
            2006: self.env.zealot_id,
            1898: self.env.stalker_id,
            1899: self.env.zealot_id,
            54: self.env.medivac_id,
            74: self.env.stalker_id,
            73: self.env.zealot_id,
            105: self.env.zergling_id,
            98: 98
        }

        # Define the set of unit types for imap(s)
        self.unitTypes = list(set(
            [self.env.agents[agent].unit_type for agent in range(
                len(self.env.agents))]
            + [self.standardEnemyUnitTypes[self.env.enemies[agent].unit_type]
                for agent in range(len(self.env.enemies))]
        ))

        # Define unit string names
        self.unitNames = {
            self.env.baneling_id: "baneling",
            self.env.colossus_id: "colossus",
            self.env.hydralisk_id: "hydralisk",
            self.env.marauder_id: "marauder",
            self.env.marine_id: "marine",
            self.env.medivac_id: "medivac",
            self.env.stalker_id: "stalker",
            self.env.zealot_id: "zealot",
            self.env.zergling_id: "zergling",
        }

        # Generally, for each agent in the array, the values are ordered:
        #   relative health, relative energy or weapon cooldown, relative x, relative y
        #   shield, type
        # Enemies are the exception in that we do not have the energy/cooldown value
        self.allyFeatureCount = 4 + self.env.shield_bits_ally + self.env.unit_type_bits
        self.enemyFeatureCount = 3 + self.env.shield_bits_enemy + self.env.unit_type_bits
        self.initializeAgentInfluenceFeatures()

    def updateInfluenceMap(self, env):
        """ Updates the influence map based on the environments current global state.
        """
        globalState = env.get_state_original()

        self.influenceMap -= self.influenceMap  # Sets everything back to 0

        for ally_id in range(len(self.allies)):
            healthIdx = ally_id * self.allyFeatureCount
            if globalState[healthIdx] > 0:
                #self.allies[ally_id]["agentInfluence"] = -globalState[healthIdx] / self.allies[ally_id]["influenceDistanceScaleMatrix"]
                self.allies[ally_id]["agentInfluence"] = -self.getInfluenceWeight(self.env.agents[ally_id]) / self.allies[ally_id]["influenceDistanceScaleMatrix"]

                self.addAgentInfluenceToInfluenceMap(
                    influenceMap=self.influenceMap,
                    agentInfluence=self.allies[ally_id]["agentInfluence"],
                    x_range=self.allies[ally_id]["x_attackRange"],
                    y_range=self.allies[ally_id]["y_attackRange"],
                    centerPositionInAgentInfluence=(
                        self.allies[ally_id]["x_attackRange"], self.allies[ally_id]["y_attackRange"]),
                    agentPositionInInfluenceMap=(round(env.agents[ally_id].pos.x * self.x_positionScale), round(
                        env.agents[ally_id].pos.y * self.y_positionScale))
                )

        enemyStartIdx = self.allyFeatureCount * len(self.allies)
        for enemy_id in range(len(self.enemies)):
            healthIdx = enemy_id * self.enemyFeatureCount + enemyStartIdx
            if globalState[healthIdx] > 0:
                self.enemies[enemy_id]["agentInfluence"] = self.getInfluenceWeight(self.env.enemies[enemy_id]) / self.enemies[enemy_id]["influenceDistanceScaleMatrix"]

                self.addAgentInfluenceToInfluenceMap(
                    influenceMap=self.influenceMap,
                    agentInfluence=self.enemies[enemy_id]["agentInfluence"],
                    x_range=self.enemies[enemy_id]["x_attackRange"],
                    y_range=self.enemies[enemy_id]["y_attackRange"],
                    centerPositionInAgentInfluence=(
                        self.allies[ally_id]["x_attackRange"], self.allies[ally_id]["y_attackRange"]),
                    agentPositionInInfluenceMap=(round(env.enemies[enemy_id].pos.x * self.x_positionScale), round(
                        env.enemies[enemy_id].pos.y * self.y_positionScale))
                )

        if self.display_image:
            self.display_influence_map()

        if self.save_image:
            self.save_influence_map_image()

    def updateEnemyInfluenceMap(self, env):
        """ Updates the influence map based on the environments current global state.
        """
        globalState = env.get_state_original()

        self.influenceMap -= self.influenceMap  # Sets everything back to 0

        for ally_id in range(len(self.allies)):
            healthIdx = ally_id * self.allyFeatureCount
            if globalState[healthIdx] > 0:
                #self.allies[ally_id]["agentInfluence"] = -globalState[healthIdx] / self.allies[ally_id]["influenceDistanceScaleMatrix"]
                self.allies[ally_id]["agentInfluence"] = self.getInfluenceWeight(self.env.agents[ally_id]) / self.allies[ally_id]["influenceDistanceScaleMatrix"]

                self.addAgentInfluenceToInfluenceMap(
                    influenceMap=self.influenceMap,
                    agentInfluence=self.allies[ally_id]["agentInfluence"],
                    x_range=self.allies[ally_id]["x_attackRange"],
                    y_range=self.allies[ally_id]["y_attackRange"],
                    centerPositionInAgentInfluence=(
                        self.allies[ally_id]["x_attackRange"], self.allies[ally_id]["y_attackRange"]),
                    agentPositionInInfluenceMap=(round(env.agents[ally_id].pos.x * self.x_positionScale), round(
                        env.agents[ally_id].pos.y * self.y_positionScale))
                )

        enemyStartIdx = self.allyFeatureCount * len(self.allies)
        for enemy_id in range(len(self.enemies)):
            healthIdx = enemy_id * self.enemyFeatureCount + enemyStartIdx
            if globalState[healthIdx] > 0:
                self.enemies[enemy_id]["agentInfluence"] = -self.getInfluenceWeight(self.env.enemies[enemy_id]) / self.enemies[enemy_id]["influenceDistanceScaleMatrix"]

                self.addAgentInfluenceToInfluenceMap(
                    influenceMap=self.influenceMap,
                    agentInfluence=self.enemies[enemy_id]["agentInfluence"],
                    x_range=self.enemies[enemy_id]["x_attackRange"],
                    y_range=self.enemies[enemy_id]["y_attackRange"],
                    centerPositionInAgentInfluence=(
                        self.allies[ally_id]["x_attackRange"], self.allies[ally_id]["y_attackRange"]),
                    agentPositionInInfluenceMap=(round(env.enemies[enemy_id].pos.x * self.x_positionScale), round(
                        env.enemies[enemy_id].pos.y * self.y_positionScale))
                )

        # inverse_map = np.zeros(
        #     (self.x_influenceMap, self.y_influenceMap))
        
        # for i in range(64):
        #     for j in range(64):
        #         inverse_map[i,j] = self.influenceMap[63-i, 63-j]
        
        # self.influenceMap = inverse_map

        if self.display_image:
            self.display_influence_map()

        if self.save_image:
            self.save_influence_map_image()

    def updateLocalInfluenceMap(self, local_obs):
        # TODO: Update this logic, as it is a hardcoded prototype
        offset = 4
        n_enemies = self.enemies.size
        n_allies = self.allies.size - 1
        n_features = 5  # +1 if any agent has a shield, +1 if multiple agent types
        if np.array([self.env.agents[agent].shield_max != 0 for agent in self.env.agents]).any() or \
                np.array([self.env.enemies[agent].shield_max != 0 for agent in self.env.enemies]).any():
            n_features += 1
        if len(self.unitTypes) > 1:
            n_features += 1
        x_offset = 2
        y_offset = 3
        healthOffset = 4
        self.enemy_offsets = []
        for agent_id, agent_obs in enumerate(local_obs):
            # Reset the agent's local observed MAIM
            self.allies[agent_id]["agentObservedLocalInfluenceMAIM"] -= self.allies[agent_id]["agentObservedLocalInfluenceMAIM"]#zeroing out
            agent_enemy_offsets = []
            for enemy_id in range(n_enemies):
                offset_x = offset + enemy_id * n_features + x_offset
                offset_y = offset + enemy_id * n_features + y_offset
                healthIdx = offset + enemy_id * n_features + healthOffset
                unitType = self.unitTypes.index(
                    self.standardEnemyUnitTypes[self.env.enemies[enemy_id].unit_type])
                agent_enemy_offsets.append({ #output
                    "x": round(self.sight_range + agent_obs[offset_x])
                    , "y": round(self.sight_range + agent_obs[offset_y])
                })
                # if enemy is visible and alive
                if agent_obs[offset + enemy_id * n_features] and agent_obs[healthIdx] > 0:
                    # Calculate agent's influence and add it to the current agent's local observations
                    self.addAgentInfluenceToInfluenceMap(
                        influenceMap=self.allies[agent_id]["agentObservedLocalInfluenceMAIM"],
                        agentInfluence=self.enemies[enemy_id]["influenceRangeFilter"] if self.booleanInfluenceMap else agent_obs[healthIdx] /
                        self.enemies[enemy_id]["influenceDistanceScaleMatrix"],
                        x_range=self.enemies[enemy_id]["x_attackRange"],
                        y_range=self.enemies[enemy_id]["y_attackRange"],
                        centerPositionInAgentInfluence=(
                            self.enemies[enemy_id]["x_attackRange"], self.enemies[enemy_id]["y_attackRange"]),
                        agentPositionInInfluenceMap=(round(self.sight_range + agent_obs[offset_x]), round(
                            self.sight_range + agent_obs[offset_y])),
                        agentLayer=unitType
                    )
            # Add current agent's enemy offset to full list
            self.enemy_offsets.append(agent_enemy_offsets)

            for ally_id in range(n_allies):
                offset_x = offset + n_enemies * n_features + ally_id * n_features + x_offset
                offset_y = offset + n_enemies * n_features + ally_id * n_features + y_offset
                healthIdx = offset + n_enemies * n_features + ally_id * n_features + healthOffset
                ally_id_adjusted = ally_id if ally_id < agent_id else ally_id + 1
                unitType = self.unitTypes.index(
                    self.env.agents[ally_id_adjusted].unit_type)
                # if ally is visible and alive
                if agent_obs[offset + n_enemies * n_features + ally_id * n_features] and agent_obs[healthIdx] > 0:
                    # Calculate agent's influence and add it to the current agent's local observations
                    self.addAgentInfluenceToInfluenceMap(
                        influenceMap=self.allies[agent_id]["agentObservedLocalInfluenceMAIM"],
                        agentInfluence=-self.allies[ally_id_adjusted]["influenceRangeFilter"] if self.booleanInfluenceMap else -
                        agent_obs[healthIdx] /
                        self.allies[ally_id_adjusted]["influenceDistanceScaleMatrix"],
                        x_range=self.allies[ally_id_adjusted]["x_attackRange"],
                        y_range=self.allies[ally_id_adjusted]["y_attackRange"],
                        centerPositionInAgentInfluence=(
                            self.allies[ally_id_adjusted]["x_attackRange"], self.allies[ally_id_adjusted]["y_attackRange"]),
                        agentPositionInInfluenceMap=(round(self.sight_range + agent_obs[offset_x]), round(
                            self.sight_range + agent_obs[offset_y])),
                        agentLayer=unitType
                    )

            # if the observing agent is alive
            if agent_obs[-(1 + n_features - 5)] > 0:
                self.addAgentInfluenceToInfluenceMap(
                    influenceMap=self.allies[agent_id]["agentObservedLocalInfluenceMAIM"],
                    agentInfluence=-self.allies[agent_id]["influenceRangeFilter"] if self.booleanInfluenceMap else -
                    agent_obs[healthIdx] /
                    self.allies[agent_id]["influenceDistanceScaleMatrix"],
                    x_range=self.allies[agent_id]["x_attackRange"],
                    y_range=self.allies[agent_id]["y_attackRange"],
                    centerPositionInAgentInfluence=(
                        self.allies[agent_id]["x_attackRange"], self.allies[agent_id]["y_attackRange"]),
                    agentPositionInInfluenceMap=(
                        round(self.sight_range), round(self.sight_range)),
                    agentLayer=self.unitTypes.index(
                        self.env.agents[agent_id].unit_type)
                )

            # pd.DataFrame(self.allies[agent_id]["agentObservedLocalInfluenceMAIM"][:, :, 0]).to_csv(
            #     "layer_0.csv", index=False)
            # pd.DataFrame(self.allies[agent_id]["agentObservedLocalInfluenceMAIM"][:, :, 1]).to_csv(
            #     "layer_1.csv", index=False)
            if self.booleanInfluenceMap and self.clipBooleanInfluenceMap:
                np.clip(self.allies[agent_id]["agentObservedLocalInfluenceMAIM"], -1,
                        1, self.allies[agent_id]["agentObservedLocalInfluenceMAIM"])
        # TODO: End todo

    def updateLocalEnemyInfluenceMap(self, local_obs):
        # TODO: Update this logic, as it is a hardcoded prototype
        offset = 4
        n_enemies = self.allies.size
        n_allies = self.enemies.size - 1
        n_features = 5  # +1 if any agent has a shield, +1 if multiple agent types
        if np.array([self.env.agents[agent].shield_max != 0 for agent in self.env.agents]).any() or \
                np.array([self.env.enemies[agent].shield_max != 0 for agent in self.env.enemies]).any():
            n_features += 1
        if len(self.unitTypes) > 1:
            n_features += 1
        x_offset = 2
        y_offset = 3
        healthOffset = 4
        self.enemy_offsets = []
        for agent_id, agent_obs in enumerate(local_obs):
            # Reset the agent's local observed MAIM
            self.enemies[agent_id]["agentObservedLocalInfluenceMAIM"] -= self.enemies[agent_id]["agentObservedLocalInfluenceMAIM"]#zeroing out
            agent_enemy_offsets = []
            for enemy_id in range(n_enemies):
                offset_x = offset + enemy_id * n_features + x_offset
                offset_y = offset + enemy_id * n_features + y_offset
                healthIdx = offset + enemy_id * n_features + healthOffset
                unitType = self.unitTypes.index(
                    self.standardEnemyUnitTypes[self.env.agents[enemy_id].unit_type])
                agent_enemy_offsets.append({ #output
                    "x": round(self.sight_range + agent_obs[offset_x])
                    , "y": round(self.sight_range + agent_obs[offset_y])
                })
                # if enemy is visible and alive
                if agent_obs[offset + enemy_id * n_features] and agent_obs[healthIdx] > 0:
                    # Calculate agent's influence and add it to the current agent's local observations
                    self.addAgentInfluenceToInfluenceMap(
                        influenceMap=self.enemies[agent_id]["agentObservedLocalInfluenceMAIM"],
                        agentInfluence=self.allies[enemy_id]["influenceRangeFilter"] if self.booleanInfluenceMap else agent_obs[healthIdx] /
                        self.allies[enemy_id]["influenceDistanceScaleMatrix"],
                        x_range=self.allies[enemy_id]["x_attackRange"],
                        y_range=self.allies[enemy_id]["y_attackRange"],
                        centerPositionInAgentInfluence=(
                            self.allies[enemy_id]["x_attackRange"], self.allies[enemy_id]["y_attackRange"]),
                        agentPositionInInfluenceMap=(round(self.sight_range + agent_obs[offset_x]), round(
                            self.sight_range + agent_obs[offset_y])),
                        agentLayer=unitType
                    )
            # Add current agent's enemy offset to full list
            self.enemy_offsets.append(agent_enemy_offsets)

            for ally_id in range(n_allies):
                offset_x = offset + n_enemies * n_features + ally_id * n_features + x_offset
                offset_y = offset + n_enemies * n_features + ally_id * n_features + y_offset
                healthIdx = offset + n_enemies * n_features + ally_id * n_features + healthOffset
                ally_id_adjusted = ally_id if ally_id < agent_id else ally_id + 1
                unitType = self.unitTypes.index(
                    self.env.enemies[ally_id_adjusted].unit_type)
                # if ally is visible and alive
                if agent_obs[offset + n_enemies * n_features + ally_id * n_features] and agent_obs[healthIdx] > 0:
                    # Calculate agent's influence and add it to the current agent's local observations
                    self.addAgentInfluenceToInfluenceMap(
                        influenceMap=self.enemies[agent_id]["agentObservedLocalInfluenceMAIM"],
                        agentInfluence=-self.enemies[ally_id_adjusted]["influenceRangeFilter"] if self.booleanInfluenceMap else -
                        agent_obs[healthIdx] /
                        self.enemies[ally_id_adjusted]["influenceDistanceScaleMatrix"],
                        x_range=self.enemies[ally_id_adjusted]["x_attackRange"],
                        y_range=self.enemies[ally_id_adjusted]["y_attackRange"],
                        centerPositionInAgentInfluence=(
                            self.enemies[ally_id_adjusted]["x_attackRange"], self.enemies[ally_id_adjusted]["y_attackRange"]),
                        agentPositionInInfluenceMap=(round(self.sight_range + agent_obs[offset_x]), round(
                            self.sight_range + agent_obs[offset_y])),
                        agentLayer=unitType
                    )

            # if the observing agent is alive
            if agent_obs[-(1 + n_features - 5)] > 0:
                self.addAgentInfluenceToInfluenceMap(
                    influenceMap=self.enemies[agent_id]["agentObservedLocalInfluenceMAIM"],
                    agentInfluence=-self.enemies[agent_id]["influenceRangeFilter"] if self.booleanInfluenceMap else -
                    agent_obs[healthIdx] /
                    self.enemies[agent_id]["influenceDistanceScaleMatrix"],
                    x_range=self.enemies[agent_id]["x_attackRange"],
                    y_range=self.enemies[agent_id]["y_attackRange"],
                    centerPositionInAgentInfluence=(
                        self.enemies[agent_id]["x_attackRange"], self.enemies[agent_id]["y_attackRange"]),
                    agentPositionInInfluenceMap=(
                        round(self.sight_range), round(self.sight_range)),
                    agentLayer=self.unitTypes.index(
                        self.env.enemies[agent_id].unit_type)
                )

            # pd.DataFrame(self.allies[agent_id]["agentObservedLocalInfluenceMAIM"][:, :, 0]).to_csv(
            #     "layer_0.csv", index=False)
            # pd.DataFrame(self.allies[agent_id]["agentObservedLocalInfluenceMAIM"][:, :, 1]).to_csv(
            #     "layer_1.csv", index=False)
            if self.booleanInfluenceMap and self.clipBooleanInfluenceMap:
                np.clip(self.allies[agent_id]["agentObservedLocalInfluenceMAIM"], -1,
                        1, self.allies[agent_id]["agentObservedLocalInfluenceMAIM"])
        # TODO: End todo

    def getInfluenceWeight(self, unit, ratio=100):
        """ calculates the weight of the influence map.
            Example: A marine has 45 health, so the influence map for a single marine would maximize to 45/100 or .45
            before being reduced by distance.

        """ 
        weight = 0
        weight += unit.health
        weight += unit.shield

        return  weight / ratio

    def initializeAgentInfluenceFeatures(self):
        self.allies = np.empty(len(self.env.agents), dtype=np.object)
        self.enemies = np.empty(len(self.env.enemies), dtype=np.object)

        for ally in range(len(self.env.agents)):
            self.allies[ally] = {
                "x_attackRange": int((self.unitAttackRanges[self.env.agents[ally].unit_type] / self.env.map_x) * self.x_influenceMap),
                "y_attackRange": int((self.unitAttackRanges[self.env.agents[ally].unit_type] / self.env.map_y) * self.y_influenceMap)
            }
            self.allies[ally]["x_agentInfluenceLimit"] = self.allies[ally]["x_attackRange"] * 2 + 1
            self.allies[ally]["y_agentInfluenceLimit"] = self.allies[ally]["y_attackRange"] * 2 + 1
            self.allies[ally]["influenceRangeFilter"] = self.getAgentInfluenceRangeFilter(
                x_limit=self.allies[ally]["x_agentInfluenceLimit"],
                y_limit=self.allies[ally]["y_agentInfluenceLimit"],
                x_attackRange=self.allies[ally]["x_attackRange"],
                y_attackRange=self.allies[ally]["y_attackRange"]
            )

            self.allies[ally]["influenceDistanceScaleMatrix"] = self.getAgentInfluenceDistanceScaleMatrix(
                influenceRangeFilter=self.allies[ally]["influenceRangeFilter"],
                x_center=self.allies[ally]["x_attackRange"],
                y_center=self.allies[ally]["y_attackRange"]
            )

            self.allies[ally]["agentInfluence"] = np.zeros(
                (self.allies[ally]["x_agentInfluenceLimit"], self.allies[ally]["y_agentInfluenceLimit"]))
            
            # For use in local observations
            self.allies[ally]["agentObservedLocalInfluenceMAIM"] = np.zeros(
                (self.sight_range * 2 + 1, self.sight_range * 2 + 1, 2))
                # (self.sight_range * 2 + 1, self.sight_range * 2 + 1, len(self.unitTypes)))

        for enemy in range(len(self.env.enemies)):
            self.enemies[enemy] = {
                "x_attackRange": int((self.unitAttackRanges[self.env.enemies[enemy].unit_type] / self.env.map_x) * self.x_influenceMap),
                "y_attackRange": int((self.unitAttackRanges[self.env.enemies[enemy].unit_type] / self.env.map_y) * self.y_influenceMap)
            }
            self.enemies[enemy]["x_agentInfluenceLimit"] = self.enemies[enemy]["x_attackRange"] * 2 + 1
            self.enemies[enemy]["y_agentInfluenceLimit"] = self.enemies[enemy]["y_attackRange"] * 2 + 1
            self.enemies[enemy]["influenceRangeFilter"] = self.getAgentInfluenceRangeFilter(
                x_limit=self.enemies[enemy]["x_agentInfluenceLimit"],
                y_limit=self.enemies[enemy]["y_agentInfluenceLimit"],
                x_attackRange=self.enemies[enemy]["x_attackRange"],
                y_attackRange=self.enemies[enemy]["y_attackRange"]
            )

            self.enemies[enemy]["influenceDistanceScaleMatrix"] = self.getAgentInfluenceDistanceScaleMatrix(
                influenceRangeFilter=self.enemies[enemy]["influenceRangeFilter"],
                x_center=self.enemies[enemy]["x_attackRange"],
                y_center=self.enemies[enemy]["y_attackRange"]
            )

            self.enemies[enemy]["agentInfluence"] = np.zeros(
                (self.enemies[enemy]["x_agentInfluenceLimit"], self.enemies[enemy]["y_agentInfluenceLimit"]))
            
            self.enemies[enemy]["agentObservedLocalInfluenceMAIM"] = np.zeros(
                (self.sight_range * 2 + 1, self.sight_range * 2 + 1, 2))

    def getAgentInfluenceRangeFilter(self, x_limit: int, y_limit: int, x_attackRange: int, y_attackRange: int) -> np.array:
        x_dim = np.arange(x_limit)[:, None]  # Makes this a column array
        y_dim = np.arange(y_limit)
        return ((x_dim-x_attackRange)/x_attackRange)**2 + ((y_dim-y_attackRange)/y_attackRange)**2 <= 1

    def getAgentInfluenceDistanceScaleMatrix(self, influenceRangeFilter: np.array, x_center: int, y_center: int) -> np.array:
        """
        Produces a matrix that is equal to 1 + the distance from the center
        for each point in the matrix that falls in the influence range, infinity otherwise.
        """
        distanceMatrix = np.full_like(influenceRangeFilter, np.inf, dtype=float)

        for x in range(distanceMatrix.shape[0]):
            for y in range(distanceMatrix.shape[1]):
                if influenceRangeFilter[x][y]:
                    distanceMatrix[x][y] = 1 + \
                         np.sqrt((x_center - x) ** 2 + (y_center - y) ** 2)

        return distanceMatrix

    def addAgentInfluenceToInfluenceMap(self,
        influenceMap: np.array,
        agentInfluence: np.array,
        x_range: int,
        y_range: int,
        centerPositionInAgentInfluence: Tuple[int, int],
        agentPositionInInfluenceMap: Tuple[int, int],
        agentLayer: int = None
    ) -> None:
        # Find the limits of the full influence map first
        # Add 1 extra because slices are upper exclusive
        pos_x_lim = min(
            agentPositionInInfluenceMap[0] + x_range + 1, influenceMap.shape[0])
        neg_x_lim = max(agentPositionInInfluenceMap[0] - x_range, 0)
        # Add 1 extra because slices are upper exclusive
        pos_y_lim = min(
            agentPositionInInfluenceMap[1] + y_range + 1, influenceMap.shape[1])
        neg_y_lim = max(agentPositionInInfluenceMap[1] - y_range, 0)

        # Now find the limits of the agent influence map
        agent_pos_x = min(
            pos_x_lim - agentPositionInInfluenceMap[0] + x_range, agentInfluence.shape[0])
        agent_neg_x = max(
            neg_x_lim - agentPositionInInfluenceMap[0] + x_range, 0)
        agent_pos_y = min(
            pos_y_lim - agentPositionInInfluenceMap[1] + y_range, agentInfluence.shape[1])
        agent_neg_y = max(
            neg_y_lim - agentPositionInInfluenceMap[1] + y_range, 0)

        # # Perform the addition to the full influence map
        # influenceMap[neg_x_lim:pos_x_lim,
        #              neg_y_lim:pos_y_lim] += agentInfluence[agent_neg_x:agent_pos_x, agent_neg_y:agent_pos_y]

        # Perform the addition to the full influence map
        if agentLayer is not None:
            influenceMap[neg_x_lim:pos_x_lim,
                         neg_y_lim:pos_y_lim,
                         agentLayer] += agentInfluence[agent_neg_x:agent_pos_x, agent_neg_y:agent_pos_y]
        else:
            influenceMap[neg_x_lim:pos_x_lim,
                         neg_y_lim:pos_y_lim] += agentInfluence[agent_neg_x:agent_pos_x, agent_neg_y:agent_pos_y]

    def display_influence_map(self, display_steps=1):
        """ Updates the image map currently being displayed.

            Args:
                display_steps (int): Every x steps the display will update.
        """

        if self.env._episode_steps % display_steps == 0:
            self.influence_map_image.set_data(self.influenceMap.T * -1.0)
            self.plt.draw()
            self.plt.pause(0.001)

    def save_influence_map_image(self, image_steps=1):
        """ Saves the current influence map to disk as an image.

            Args:
                image_steps (int): Every x steps an image will be saved.

        """

        episodes = self.env._episode_count
        steps = self.env._episode_steps

        if self.env._episode_steps % image_steps == 0:
            print('----------------------------------------------------------')
            plt.figure(figsize=(10, 10))
            plt.imshow(self.influenceMap.T * -1.0, cmap="PiYG", interpolation="nearest")
            plt.clim(-3, 3)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f'ep_{episodes}_step_{steps}_mappo.png', dpi=150)
            plt.close()
    
    def save_enemy_influence_map_image(self, image_steps=1):
        """ Saves the current influence map to disk as an image.

            Args:
                image_steps (int): Every x steps an image will be saved.

        """

        episodes = self.env._episode_count
        steps = self.env._episode_steps

        if self.env._episode_steps % image_steps == 0:
            print('----------------------------------------------------------')
            plt.figure(figsize=(10, 10))
            plt.imshow(self.influenceMap.T * -1.0, cmap="PiYG", interpolation="nearest")
            plt.clim(-3, 3)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f'ep_{episodes}_step_{steps}_enemy_mappo.png', dpi=150)
            plt.close()

    def get_size(self):
        return self.x_influenceMap * self.y_influenceMap

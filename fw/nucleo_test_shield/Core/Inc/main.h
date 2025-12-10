/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

void HAL_TIM_MspPostInit(TIM_HandleTypeDef *htim);

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define ADC_TSENS0_Pin GPIO_PIN_0
#define ADC_TSENS0_GPIO_Port GPIOC
#define ADC_TSENS1_Pin GPIO_PIN_1
#define ADC_TSENS1_GPIO_Port GPIOC
#define ADC_TSENS2_Pin GPIO_PIN_2
#define ADC_TSENS2_GPIO_Port GPIOC
#define ADC_IGEN_Pin GPIO_PIN_3
#define ADC_IGEN_GPIO_Port GPIOC
#define ADC_VIN_RAW_Pin GPIO_PIN_0
#define ADC_VIN_RAW_GPIO_Port GPIOA
#define ADC_VIN_AMP_Pin GPIO_PIN_1
#define ADC_VIN_AMP_GPIO_Port GPIOA
#define USART_TX_Pin GPIO_PIN_2
#define USART_TX_GPIO_Port GPIOA
#define USART_RX_Pin GPIO_PIN_3
#define USART_RX_GPIO_Port GPIOA
#define DAC_BIAS_Pin GPIO_PIN_4
#define DAC_BIAS_GPIO_Port GPIOA
#define DAC_IGEN_Pin GPIO_PIN_5
#define DAC_IGEN_GPIO_Port GPIOA
#define UI_D0_Pin GPIO_PIN_6
#define UI_D0_GPIO_Port GPIOA
#define UI_D1_Pin GPIO_PIN_7
#define UI_D1_GPIO_Port GPIOA
#define UI_D2_Pin GPIO_PIN_4
#define UI_D2_GPIO_Port GPIOC
#define UI_D3_Pin GPIO_PIN_5
#define UI_D3_GPIO_Port GPIOC
#define UI_C0_Pin GPIO_PIN_10
#define UI_C0_GPIO_Port GPIOB
#define UI_C1_Pin GPIO_PIN_12
#define UI_C1_GPIO_Port GPIOB
#define UI_C2_Pin GPIO_PIN_13
#define UI_C2_GPIO_Port GPIOB
#define PU_NTC3_Pin GPIO_PIN_14
#define PU_NTC3_GPIO_Port GPIOB
#define PU_NTC2_Pin GPIO_PIN_15
#define PU_NTC2_GPIO_Port GPIOB
#define PU_NTC1_Pin GPIO_PIN_6
#define PU_NTC1_GPIO_Port GPIOC
#define PU_NTC0_Pin GPIO_PIN_7
#define PU_NTC0_GPIO_Port GPIOC
#define PU_RES5_Pin GPIO_PIN_9
#define PU_RES5_GPIO_Port GPIOC
#define PU_RES4_Pin GPIO_PIN_8
#define PU_RES4_GPIO_Port GPIOA
#define PU_RES3_Pin GPIO_PIN_9
#define PU_RES3_GPIO_Port GPIOA
#define PU_RES2_Pin GPIO_PIN_10
#define PU_RES2_GPIO_Port GPIOA
#define PU_RES1_Pin GPIO_PIN_11
#define PU_RES1_GPIO_Port GPIOA
#define PU_RES0_Pin GPIO_PIN_12
#define PU_RES0_GPIO_Port GPIOA
#define TMS_Pin GPIO_PIN_13
#define TMS_GPIO_Port GPIOA
#define TCK_Pin GPIO_PIN_14
#define TCK_GPIO_Port GPIOA
#define TIM2_HEAT_Pin GPIO_PIN_15
#define TIM2_HEAT_GPIO_Port GPIOA
#define SWO_Pin GPIO_PIN_3
#define SWO_GPIO_Port GPIOB
#define PD_DIODE_Pin GPIO_PIN_4
#define PD_DIODE_GPIO_Port GPIOB
#define PD_BJT_Pin GPIO_PIN_5
#define PD_BJT_GPIO_Port GPIOB
#define PD_IGEN_Pin GPIO_PIN_6
#define PD_IGEN_GPIO_Port GPIOB
#define PD_RES_Pin GPIO_PIN_7
#define PD_RES_GPIO_Port GPIOB

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */

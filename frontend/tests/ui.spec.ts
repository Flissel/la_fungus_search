import { test, expect } from '@playwright/test'

test.describe('EmbeddingGemma React UI', () => {
  test('renders sidebar and live chart', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByText('Fungus (MCMP) Frontend')).toBeVisible()
    await expect(page.getByText('Live pheromone network')).toBeVisible()
    await expect(page.getByLabel('Query')).toBeVisible()
    await expect(page.getByText('Viz dims')).toBeVisible()
  })

  test('apply changes and interact', async ({ page }) => {
    await page.goto('/')
    await page.getByLabel('Query').fill('test query')
    await page.getByRole('button', { name: 'Start' }).click()
    await page.getByRole('button', { name: 'Apply' }).click()
    await expect(page.getByText('Live pheromone network')).toBeVisible()
  })
})



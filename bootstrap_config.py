#!/usr/bin/env python
"""
Phase 1 Bootstrap: Creates config directory and initial YAML files.
Run: python bootstrap_config.py
"""

import os
import yaml
from pathlib import Path

# Get repo root
REPO_ROOT = Path(__file__).parent
CONFIG_DIR = REPO_ROOT / "config"
APP_CONFIG_DIR = REPO_ROOT / "app" / "config"

# Create directories
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Created: {CONFIG_DIR}")
print(f"Created: {APP_CONFIG_DIR}")

# Create services.yaml
services_yaml = """# Service Registry Configuration
# Phase 1 Config Layer: Replaces hardcoded _slug_intent() and _FLOWS registry

services:
  tu_application:
    name: "Заявление на технические условия"
    name_kz: "Техникалық шарттар бойынша өтініш"
    code: "tu"
    slug: "tu_application"
    type: "linear"
    page_context: "технических условий"
    
  real_estate:
    name: "Добавление объекта недвижимости"
    name_kz: "Жылжымайтын мүлік объектісін қосу"
    code: "re"
    slug: "real_estate"
    type: "scenario"
    page_context: "объект недвижимости"
    
  supply_contract:
    name: "Договор электроснабжения"
    name_kz: "Электрмен жабдықтау шарты"
    code: "sc"
    slug: "supply_contract"
    type: "faq"
    page_context: "договор"
    
  load_calculation:
    name: "Расчёт электрической нагрузки"
    name_kz: "Электр жүктемесінің есебі"
    code: "lc"
    slug: "load_calculation"
    type: "faq"
    page_context: "расчет"
    
  draft_design:
    name: "Разработка эскизного проекта"
    name_kz: "Эскиздік жобаны әзірлеу"
    code: "dd"
    slug: "draft_design"
    type: "faq"
    page_context: "эскизный"
    
  construction_works:
    name: "Строительно-монтажные работы"
    name_kz: "Құрылыс-монтаж жұмыстары"
    code: "cw"
    slug: "construction_works"
    type: "faq"
    page_context: "смр"
    
  meter_sealing:
    name: "Установка/снятие пломбы"
    name_kz: "Пломбаны орнату/алу"
    code: "ms"
    slug: "meter_sealing"
    type: "faq"
    page_context: "пломб"
    
  general:
    name: "General FAQ"
    name_kz: "Жалпы ССҚ"
    code: "gen"
    slug: "general"
    type: "faq"
    page_context: null
"""

with open(CONFIG_DIR / "services.yaml", "w", encoding="utf-8") as f:
    f.write(services_yaml)
print(f"Created: {CONFIG_DIR / 'services.yaml'}")

# Create __init__.py files
for dir_path in [CONFIG_DIR, APP_CONFIG_DIR]:
    init_file = dir_path / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# Config package\n")
        print(f"Created: {init_file}")

print("✓ Bootstrap complete!")

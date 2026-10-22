import { useEffect, useState } from 'react'
import { fetchJson } from '../api'
import type { PersonalProfile } from '../types'
import { RATING_TIERS } from '../ratingTier.mjs'

export default function MethodPage() {
  const [personal, setPersonal] = useState<PersonalProfile | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    const request = new AbortController()
    void fetchJson<{ model: { personal: PersonalProfile | null } }>('/api/holistic/method', { signal: request.signal }).then(data => setPersonal(data.model.personal)).catch(e => { if (!request.signal.aborted) setError(e.message) })
    return () => request.abort()
  }, [])
  return <section className="method-page"><span className="eyebrow">mogzilla / behind the rate</span><h1>За цифрами.<br />Как это работает.</h1><p className="section-intro">Рейт по шкале mogzilla с категориями looksmaxing, разбор пропорций и проверка качества снимка.</p>
    <div className="method-grid"><article><span className="method-number">01</span><h2>Данные и разметка</h2><p>Рейт рассчитывается по изображению. При подготовке текущей версии использованы SCUT-FBP5500, фотографии из MEBeauty и дополнительные подборки портретов с ручными оценками.</p><p>Шкала mogzilla ориентирована на эстетику looksmaxing-сообщества: от sub 3 до true adam. Это интерпретация внешности на конкретном снимке; единого объективного стандарта красоты нет.</p><p>Для анализа используются фрагмент лица и более широкий фрагмент головы, включая волосы.</p></article>
      <article><span className="method-number">02</span><h2>Разбор областей</h2><p>На фотографии определяются 478 ориентиров. По ним рассчитываются наклоны обоих глаз, расстояния, пропорции и различия парных точек. У каждого измерения есть формула.</p><p>На отдельных фотографиях показаны глаза, брови, челюсть и другие зоны. Разметка описывает геометрию снимка; она не является картой влияния черт на решение нейросети.</p></article>
      <article><span className="method-number">03</span><h2>Качество снимка</h2><p>Проверяются число лиц, размер, обрезанный контур, резкость, свет, положение головы и закрытые глаза. При выраженных проблемах лучше выбрать другое фото.</p><p>Три рамки кадрирования и отражение позволяют оценить чувствительность обработки. Разброс на экране относится к одному снимку; это не вероятность точности и не проверка разных фотографий человека.</p><p>Профиль и трёхмерная форма костей не оцениваются.</p></article>
      <article><span className="method-number">04</span><h2>Проверка качества</h2>{personal ? <><p>В обучающей части подборки — <b>{personal.fit_images}</b> фотографий с ручной разметкой. Ещё <b>{personal.test_images}</b> снимков отложены для проверки. Средняя абсолютная ошибка на них — <b>{personal.validation.personal.mae.toFixed(2)}</b> балла из 10; ошибка не больше балла — на <b>{Math.round(personal.validation.personal.within_one*100)}%</b> снимков.</p><p>Часть проверочных результатов уже изучалась при выборе версии. Явные повторы людей сгруппированы визуально, но полная независимость по людям не установлена. Результат требует подтверждения на новых лицах и фотографиях.</p></> : <p>{error || 'Загружаем результаты проверок…'}</p>}</article></div>
    <details className="method-footnote">
      <summary>Источники, условия применения и фотографии</summary>
      <p><b>SCUT-FBP5500</b> — исходный набор фотографий с человеческими оценками, на которых основана визуальная модель. Оценки отражают взгляды участников набора. <a href="https://github.com/HCIILAB/SCUT-FBP5500-Database-Release" target="_blank" rel="noreferrer">Датасет и условия ↗</a></p>
      <p><b>MEBeauty</b> — фотографии, вошедшие в подборку для шкалы mogzilla. Для них использованы ручные оценки модельных скаутов, а не исходные средние баллы набора. <a href="https://huggingface.co/datasets/dr-irina-lebedeva/MEBeauty-Facial-Beauty-Prediction" target="_blank" rel="noreferrer">Датасет и условия ↗</a></p>
      <p><b>Дополнительные портреты, включая digitals</b> — отдельные подборки с ручными оценками. Это дополнительные фотографии, а не единый опубликованный датасет.</p>
      <p>SCUT-FBP5500 и MEBeauty используются в рамках некоммерческого исследования с учётом условий источников.</p>
      <p>Фотографии обрабатываются в оперативной памяти сервера. Внешние AI API, автоматическое сохранение фотографий и распознавание личности не используются.</p>
    </details>
    <details className="method-footnote">
      <summary>Данные предыдущих экспериментов и проверок</summary>
      <p><b>WFLW</b> использовался для обучения прежних моделей ориентиров и технического сравнения точности измерений текущего подхода. <a href="https://github.com/wywu/LAB" target="_blank" rel="noreferrer">Источник ↗</a></p>
      <p><b>AFW / 300-W</b> — небольшой первоначальный эксперимент с 68 ориентирами; использовалась часть AFW. <a href="https://ibug.doc.ic.ac.uk/resources/300-W/" target="_blank" rel="noreferrer">Источник ↗</a></p>
      <p><b>WIDER FACE</b> — дополнительные фотографии для обучения поиска лица в прежней версии. <a href="http://shuoyang1213.me/WIDERFACE/" target="_blank" rel="noreferrer">Источник ↗</a></p>
      <p>Эти наборы содержат разметку ориентиров или рамок лица. Они использовались для локализации и её проверки, а не как источники оценок внешности для текущего рейта.</p>
    </details>
    <details className="method-footnote"><summary>Шкала категорий</summary><p>Названия обозначают интервалы итогового балла. На общей границе начинается следующая категория. Используется тот же округлённый балл, который показан в результате.</p><dl className="tier-legend">{RATING_TIERS.map(tier => <div key={tier.label}><dt>{tier.label}</dt><dd>{tier.low.toFixed(1)}–{tier.high.toFixed(1)}</dd></div>)}</dl></details>
  </section>
}

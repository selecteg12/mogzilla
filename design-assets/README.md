# mogzilla mascot

Создан встроенным инструментом imagegen 21 октября 2026 года по приложенному пользователем референсу. PNG с прозрачным фоном: `mogzilla-mascot.png`. Рабочая копия с сохранённой альфой и сжатием WebP: `../frontend/src/assets/mogzilla-mascot.webp`. Vite включает её в `/assets`, которые обслуживает локальный сервер с последней персональной моделью на 8008.

## Финальный промпт

Use case: stylized-concept. Asset type: transparent hero mascot for the website mogzilla. The attached image is a style and subject reference, not a real person. Create one polished sculptural 3D floating head inspired by this exaggerated laughing male emoji mask: bald golden-yellow glossy ceramic head, angular strong cheekbones and jaw, thick raised glossy black eyebrows, laughing closed eyes, wide playful smile, two large translucent deep cyan-blue glass teardrops at the outer eyes. Retain the striking humorous high-fashion meme quality, clean smooth ceramic and glass surfaces, crisp studio reflections. View almost front on, slight natural three-quarter turn, full isolated head entirely in frame with generous transparent padding, no neck or body, no hands, no props. Actual alpha transparent background, no backdrop, no cast floor shadow. No text, no wordmark, no watermark. Intended as a large editorial website mascot composited over orange and lavender geometry; do not draw the website or any graphic panels.

## Эмоции результата — 22 октября 2026

Исходники с альфой сохранены в `mascot-emotions/`: `calm.png`, `shocked.png`, `respect-v2.png`. Первая версия уважения `respect.png` оставлена как исходник; в приложении используется версия с очками. Рабочие WebP 640×640 находятся в `../frontend/src/assets/mascot/`. Альфа сохранена, без дорисовки кодом.

Все варианты получены встроенным imagegen с исходным маскотом как референсом. Общие ограничения: то же золотое глянцевое керамическое лицо, сильная челюсть, лысая голова, чёрные брови, тот же студийный свет; целая изолированная голова на настоящем прозрачном фоне, без шеи, тела, текста и водяных знаков.

Изменения выражений: calm — relaxed visible eyes, neutral brows, closed mouth with a subtle calm smile, no tears; shocked — raised brows, wide-open eyes and round open mouth in total astonishment, no tears; respect-v2 — BIG WIDE TOOTHY HAPPY SMILE, stylish BLACK SUNGLASSES with dark opaque lenses and a chunky classic frame, confident approving reaction, no tears. Вторая версия уважения создана по явной правке пользователя.

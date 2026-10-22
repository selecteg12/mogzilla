"""Install the local photo labeler without changing photographs or browser labels."""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'tools/preference-labeler'


def install(directory: Path) -> dict:
    directory = directory.resolve()
    if not directory.is_dir():
        raise ValueError('The dataset directory must already exist.')
    manifest_file = directory / 'manifest.json'
    manifest = json.loads(manifest_file.read_text(encoding='utf-8')) if manifest_file.exists() else []
    if not isinstance(manifest, list):
        raise ValueError('Expected the original collection manifest list, not an exported dataset ZIP.')
    records = []
    for order, row in enumerate(manifest):
        path = directory / row['path']
        path = path.resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f'Missing or unsafe image path for {row.get("id")}')
        reference = row.get('origin') == 'user_archive'
        score = row.get('user_score') if reference else None
        if score is not None and (type(score) is not int or not 1 <= score <= 10):
            raise ValueError('Invalid seed score.')
        records.append({
            'id': row['id'], 'filename': path.name, 'assetPath': path.relative_to(directory).as_posix(),
            'sha256': row['sha256'], 'size': path.stat().st_size,
            'width': row.get('width'), 'height': row.get('height'), 'order': order,
            'reference': reference, 'status': 'rated' if reference else 'pending',
            'score': score, 'scoreSource': 'user_archive' if reference else None,
            'suggestedScore': row.get('suggested_score'), 'sourcePage': row.get('source_page'),
            'usageRights': row.get('usage_rights'), 'origin': row.get('origin'), 'updatedAt': None,
        })
    ignored = []
    audit = directory / 'Проверка-C012.json'
    if audit.exists():
        note = json.loads(audit.read_text(encoding='utf-8'))
        ignored = [{'id': note['id'], 'reviewed_at_to_ignore': note['reviewed_at_to_ignore']}]
    config = {'version': 2, 'records': records, 'ignoredLegacyRatings': ignored}
    assets = directory / 'reviewer'
    assets.mkdir(exist_ok=True)
    for name in ['app.mjs', 'core.mjs', 'storage.mjs', 'style.css']:
        shutil.copyfile(APP / name, assets / name)
    (assets / 'vendor').mkdir(exist_ok=True)
    for name in ['fflate.mjs', 'fflate-LICENSE.txt', 'provenance.json']:
        shutil.copyfile(APP / 'vendor' / name, assets / 'vendor' / name)
    target = directory / 'Просмотр.html'
    backup = directory / 'Просмотр-v1.html'
    if target.exists() and not backup.exists():
        shutil.copyfile(target, backup)
    (directory / 'review-data.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copyfile(APP / 'index.html', target)
    (directory / 'КАК_РАЗМЕЧАТЬ.txt').write_text(
        'FaceLab — добавление фотографий и экспорт\n\n'
        '1. Откройте текущий просмотрщик через локальный сервер (обычно http://127.0.0.1:8002/Просмотр.html). После обновления версии перезагрузите страницу.\n'
        '2. Нажмите «Добавить фотографии», выберите сразу несколько файлов или перетащите их в область добавления. Кнопка «Выбрать папку» добавляет фотографии из папки.\n'
        '3. Ставьте баллы клавишами 1–9; 0 означает 10. Пробел пропускает снимок; Backspace отменяет действие. Можно снять оценку или исключить снимок.\n'
        '4. Фотографии и оценки автоматически сохраняются в IndexedDB этого браузера. Точные повторы файлов пропускаются. Черновые предложения не превращаются в личные оценки автоматически.\n'
        '5. Нажмите «Скачать ZIP: фото и оценки». Один архив содержит фотографии, ratings.csv и manifest.json. Передайте архив для подготовки обучения. Неразмеченные и исключённые фотографии явно отделены от подтверждённой разметки.\n'
        '6. Кнопка «Восстановить из ZIP» возвращает экспортированную подборку с фотографиями и оценками. При совпадениях сохраняется более свежая разметка.\n'
        '7. Данные привязаны к адресу страницы и браузеру. Открывайте тот же адрес 127.0.0.1:8002. ZIP — переносимая резервная копия; очистка данных браузера удалит его локальное хранилище.\n'
        '8. Текущие оценки старого просмотрщика автоматически переносятся при обновлении. Оригинальные фотографии на диске не изменяются. Передача в сторонние сервисы и обучение автоматически не выполняются.\n',
        encoding='utf-8-sig',
    )
    return {'directory': str(directory), 'builtin_photos': len(records), 'user_ratings_on_disk': sum(r['reference'] for r in records), 'browser_progress': 'migrated on page load'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.directory), ensure_ascii=False, indent=2))

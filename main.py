from nlp_model import NLPModel
from document import DocumentProcessor

def main():
    course_work_path = "course_works/test.docx"
    document_processor = DocumentProcessor(course_work_path)
    full_text = document_processor.get_text()

    nlp_model = NLPModel()
    
    processed_text = nlp_model.process_text(full_text)

    raw_query = "Базы данных"
    print(f"Исходный запрос: {raw_query}")

    processed_query = ' '.join(nlp_model.process_text(raw_query))
    print(f"Обработанный запрос: {processed_query}")

    total_similarity = nlp_model.calculate_similarity(' '.join(processed_text), processed_query)
    print(f"Релевантность всего текста: {total_similarity}")

    snippets = document_processor.get_snippets(processed_query)
    print("\nСниппеты:")
    for idx, snippet in enumerate(snippets, 1):
        snippet_similarity = nlp_model.calculate_similarity(' '.join(nlp_model.process_text(snippet)), processed_query)
        print(f"\nСниппет {idx}:\n{snippet}")
        print(f"Релевантность сниппета: {snippet_similarity}")

    if snippets:
        average_snippet_similarity = sum([nlp_model.calculate_similarity(' '.join(nlp_model.process_text(snippet)), processed_query) for snippet in snippets]) / len(snippets)
        print(f"\nОбщая оценка усредненная по всем сниппетам: {average_snippet_similarity}")
    else:
        print("\nНет сниппетов, удовлетворяющих запросу.")

if __name__ == "__main__":
    main()

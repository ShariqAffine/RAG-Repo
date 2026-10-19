from app.rag.chunker import recursive_character_chunks, table_rows_to_chunks


def test_chunker_keeps_text_and_limits_chunk_size():
    text = ("A paragraph with a sentence.\n\n" * 80).strip()
    chunks = recursive_character_chunks(text, chunk_size=180, overlap=20)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 180 for chunk in chunks)
    assert "A paragraph" in " ".join(chunks)


def test_table_chunk_repeats_header():
    rows = [["Region", "2024"], ["Africa", "2914"], ["South Asia", "3143"], ["Total", "37568"]]
    chunks = table_rows_to_chunks(rows, rows_per_chunk=1)
    assert len(chunks) == 3
    assert all("TABLE HEADERS: Region | 2024" in chunk for chunk in chunks)

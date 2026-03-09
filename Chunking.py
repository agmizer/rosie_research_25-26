#from spire.presentation import *
from pypdf import PdfReader


class Chunking:

    """
    This class is used to extract text from powerpoints, pdfs, and also has a chunking method
    """


    def extract_text_from_pdf(path_to_pdf):
        reader = PdfReader(path_to_pdf)

        pages = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)

        # this returns a single string for all the documents
        # maybe look at langchain pdf reader so it is read as one document instead of as a string
            # this helps keep tables and figures together also
        return "\n".join(pages)  


    
    
    def extract_text_from_shape(shape):
        """
        Extract text from an individual shape or grouped shapes.
        """
        text = ""
        if isinstance(shape, IAutoShape):
            if shape.TextFrame is not None:
                for para in shape.TextFrame.Paragraphs:
                    text += para.Text + "\n"
        elif isinstance(shape, GroupShape):
            for sub_shape in shape.Shapes:
                text += extract_text_from_shape(sub_shape)
        return text

    

    def extract_text_from_tables(ppt_path, output_txt_path=None):
        """
        Extract text from all table shapes in all slides of the presentation.
        """
        presentation = Presentation()
        presentation.LoadFromFile(ppt_path)

        all_text = ""
        for slide_index, slide in enumerate(presentation.Slides):
            slide_text = f"--- Slide {slide_index + 1} (Tables) ---\n"
            has_table = False
            for shape in slide.Shapes:
                if isinstance(shape, ITable):
                    has_table = True
                    for row in shape.TableRows:
                        for cell in row:
                            if cell.TextFrame is not None:
                                for para in cell.TextFrame.Paragraphs:
                                    slide_text += para.Text + "\t"
                        slide_text += "\n"
            if has_table:
                all_text += slide_text + "\n"

        if output_txt_path:
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(all_text)

        return all_text


    
    def extract_text_from_shapes(ppt_path, output_txt_path=None):
        """
        Extract text from all individual and grouped shapes in all slides of the presentation.
        """
        presentation = Presentation()
        presentation.LoadFromFile(ppt_path)

        all_text = ""
        for index, slide in enumerate(presentation.Slides):
            all_text += f"--- Slide {index + 1} ---\n"
            for shape in slide.Shapes:
                all_text += extract_text_from_shape(shape)
            all_text += "\n"  # Blank line after each slide

        if output_txt_path:
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(all_text)

        return all_text
    

    
    def extract_speaker_notes(ppt_path, output_txt_path=None):
        """
        Extract speaker notes from all slides in the presentation.
        """
        ppt = Presentation()
        ppt.LoadFromFile(ppt_path)

        all_notes = ""

        for index, slide in enumerate(ppt.Slides):
            notes_slide = slide.NotesSlide
            if notes_slide is not None and notes_slide.NotesTextFrame is not None:
                notes_text = notes_slide.NotesTextFrame.Text
                all_notes += f"--- Slide {index + 1} (Speaker Notes) ---\n"
                all_notes += notes_text + "\n\n"

        if output_txt_path:
            with open(output_txt_path, "w", encoding="utf-8") as file:
                file.write(all_notes)

        ppt.Dispose()
        return all_notes



    def extract_text_from_smartart(ppt_path, output_txt_path=None):
        """
        Extract text from all SmartArt graphics in all slides.
        """
        presentation = Presentation()
        presentation.LoadFromFile(ppt_path)

        all_text = ""
    
        for slide_index, slide in enumerate(presentation.Slides):
            slide_text = f"--- Slide {slide_index + 1} (SmartArt) ---\n"
            has_smartart = False
            for shape in slide.Shapes:
                if isinstance(shape, ISmartArt):
                    has_smartart = True
                    for node in shape.Nodes:
                        if node.TextFrame is not None:
                            slide_text += node.TextFrame.Text + "\n"
            if has_smartart:
                all_text += slide_text + "\n"
    
        if output_txt_path:
            with open(output_txt_path, "w", encoding="utf-8") as file:
                file.write(all_text)

        presentation.Dispose()
        return all_text




     #code for chunking text into managable pieces
    def chunk_text(text: str, max_words: int = 200, overlap: int = 50) -> list[str]:
        words, chunks, i = text.split(), [], 0
        while i < len(words):
            end = min(i + max_words, len(words))
            chunks.append(" ".join(words[i:end]))
            i += max_words - overlap
        return chunks

def main():
    print(Chunking.extract_text_from_pdf("pdfs/Abstract Algebra Syllabus Spring 2026.pdf"))

if __name__ == "__main__":
    main()
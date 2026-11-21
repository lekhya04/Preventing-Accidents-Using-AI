# Publishing SafeDrive-Sim on IEEE DataPort (one dataset, whole team)

Upload folder: `0_SafeDrive-Sim_COMBINED_upload_this/`. The folders 1_, 2_, 3_ are optional separate packages (only needed if you decide to publish three datasets instead of one).
All data is SYNTHETIC - keep that word in the title, abstract and documents exactly as written.

## Who uploads?
One team member uploads. DataPort's form asks for author names only "for the purposes of citation" (first name, last name, optional ORCID, optional affiliation), so the other two do not need an IEEE account to be listed as authors.
Before uploading, agree as a team on:
1. Who uploads (needs a free IEEE account; the dataset is created under that account).
2. Author order (the order you type them is the citation order) and each person's surname.
3. Whether your faculty guide is listed as an author, and whether the department pays if you choose Open Access.

## Step by step
1. Go to ieee-dataport.org and log in (create a free IEEE account if needed).
2. Click **SUBMIT A DATASET** at the top of the home page and pick **Standard** (free) or **Open Access** (one-time fee, listed as US$1,950 when checked - confirm on the site). Both give a DOI.
3. **Metadata section** - copy each field from `FORM_FILL_IN_SHEET.md`: Dataset Authors (one row each), Title, Category, Keywords, Data Format, License, Dataset Image (`SafeDrive-Sim_cover.png`), Abstract.
4. **Dataset section** - upload `SafeDrive-Sim_data.zip` and `SafeDrive-Sim_documentation.pdf` under Dataset Files, and the three scripts in `scripts/` under Scripts.
5. Tick the Terms of Use box and submit. The DOI appears at the top of the dataset page.
6. Open the published page and check title, authors, abstract, image and file list. Send the DOI to the whole team and put it in your report and paper.

## Before you click submit
- Replace every `[surname]` in the sheet; the PDF and the sheet say "[surname]" and "[assigned by IEEE DataPort after publishing]" - the PDF citation line is fine to leave until you have the DOI.
- Do not remove the word SYNTHETIC from anything.
- Standard datasets: if files are not uploaded in time DataPort may unpublish the entry, so upload the files in the same sitting.

## What to say in the report / viva
- The datasets are synthetic by design; generators, seed and assumptions are published, so anyone can regenerate them exactly.
- Accuracy numbers come from controlled synthetic benchmarks, not real roads - state this under Limitations.
- Real-world validation (real phone-sensor trips, an ethics-approved alcohol-sensor study, real crash records) is future work.

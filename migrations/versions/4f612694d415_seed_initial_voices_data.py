"""seed initial voices data

Revision ID: 4f612694d415
Revises: 3cc4cefa49b4
Create Date: 2026-01-15 10:44:33.510843

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4f612694d415'
down_revision: Union[str, Sequence[str], None] = '3cc4cefa49b4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Get table object for bulk insert
    voice_table = sa.table(
        'voices',
        sa.column('id', sa.String),
        sa.column('filename', sa.String),
        sa.column('language', sa.String),
        sa.column('region', sa.String),
        sa.column('gender', sa.String),
        sa.column('description', sa.String),
        sa.column('owner_id', sa.Integer)
    )

    op.bulk_insert(voice_table, [
        {"id": "female_english", "filename": "female_english.wav", "language": "en", "region": "US", "gender": "female", "description": "Default English Female", "owner_id": None},
        {"id": "male_english", "filename": "male_english.wav", "language": "en", "region": "US", "gender": "male", "description": "Default English Male", "owner_id": None},
        {"id": "agus", "filename": "agus.wav", "language": "es", "region": "CL", "gender": "male", "description": "Spanish Male Voice (Chile)", "owner_id": 1},
        {"id": "daniela", "filename": "daniela.wav", "language": "es", "region": "AR", "gender": "female", "description": "Piper Voice (high)", "owner_id": None},
        {"id": "carlfm", "filename": "carlfm.wav", "language": "es", "region": "ES", "gender": "male", "description": "Piper Voice (x_low)", "owner_id": None},
        {"id": "davefx", "filename": "davefx.wav", "language": "es", "region": "ES", "gender": "male", "description": "Piper Voice (medium)", "owner_id": None},
        {"id": "mls_10246", "filename": "mls_10246.wav", "language": "es", "region": "ES", "gender": "male", "description": "Piper Voice (low)", "owner_id": None},
        {"id": "mls_9972", "filename": "mls_9972.wav", "language": "es", "region": "ES", "gender": "female", "description": "Piper Voice (low)", "owner_id": None},
        {"id": "sharvard", "filename": "sharvard.wav", "language": "es", "region": "ES", "gender": "male", "description": "Piper Voice (medium)", "owner_id": None},
        {"id": "ald", "filename": "ald.wav", "language": "es", "region": "MX", "gender": "male", "description": "Piper Voice (medium)", "owner_id": None},
        {"id": "claude", "filename": "claude.wav", "language": "es", "region": "MX", "gender": "male", "description": "Piper Voice (high)", "owner_id": None}
    ])


def downgrade() -> None:
    # Remove the seeded data
    voice_ids = [
        'female_english', 'male_english', 'agus', 'daniela', 'carlfm', 
        'davefx', 'mls_10246', 'mls_9972', 'sharvard', 'ald', 'claude'
    ]
    bind = op.get_bind()
    bind.execute(
        sa.text("DELETE FROM voices WHERE id IN :ids"),
        {"ids": tuple(voice_ids)}
    )

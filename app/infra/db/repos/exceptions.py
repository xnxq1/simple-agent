import re
from functools import wraps

from sqlalchemy.exc import (
    DBAPIError,
    IntegrityError,
    NoResultFound,
    SQLAlchemyError,
)


class EntityNotFoundError(Exception):
    def __init__(self, entity_name: str, entity_id: str):
        self.entity_name = entity_name
        self.entity_id = entity_id
        super().__init__(f"{entity_name} with id={entity_id} not found")


class EntityAlreadyExistsError(Exception):
    def __init__(self, entity_name: str, field: str, value: str):
        self.entity_name = entity_name
        self.field = field
        self.value = value
        super().__init__(f"{entity_name} with {field}={value} already exists")


class ForeignKeyViolationError(Exception):
    def __init__(self, entity_name: str, referenced_entity: str):
        self.entity_name = entity_name
        self.referenced_entity = referenced_entity
        super().__init__(
            f"Cannot modify {entity_name}: referenced by {referenced_entity}"
        )


class DatabaseError(Exception):
    def __init__(self, message: str, original_error: Exception | None = None):
        self.original_error = original_error
        super().__init__(message)


class ValidationError(Exception):
    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(f"Validation error for {field}: {message}")


def map_db_error(error: Exception, entity_name: str) -> Exception:
    if isinstance(error, IntegrityError):
        error_msg = str(error.orig).lower()

        if "unique constraint" in error_msg or "duplicate key" in error_msg:
            match = re.search(r"Key \((\w+)\)=\((.+?)\)", str(error.orig))
            if match:
                field = match.group(1)
                value = match.group(2)
                return EntityAlreadyExistsError(entity_name, field, value)
            return EntityAlreadyExistsError(entity_name, "field", "value")

        if "foreign key constraint" in error_msg or "violates foreign key" in error_msg:
            match = re.search(r'table "(\w+)"', str(error.orig))
            referenced = match.group(1) if match else "related entity"
            return ForeignKeyViolationError(entity_name, referenced)

        if "not null constraint" in error_msg or "null value" in error_msg:
            match = re.search(r'column "(\w+)"', str(error.orig))
            field = match.group(1) if match else "field"
            return DatabaseError(f"Required field '{field}' cannot be null")

        return DatabaseError(f"Data integrity violation: {error.orig}")

    if isinstance(error, NoResultFound):
        return EntityNotFoundError(entity_name, "unknown")

    if isinstance(error, DBAPIError):
        return DatabaseError(f"Database connection error: {error.orig}")

    if isinstance(error, SQLAlchemyError):
        return DatabaseError(f"Database error: {str(error)}")

    return DatabaseError(f"Unexpected database error: {str(error)}", error)


def handle_db_errors(func):
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        entity_name = self.db_entity.name.capitalize().rstrip("s")

        try:
            return await func(self, *args, **kwargs)
        except Exception as e:
            domain_error = map_db_error(e, entity_name)

            raise domain_error from e

    return wrapper
